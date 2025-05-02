if __name__ == "__main__":
    #data loading
    from torchvision import datasets,transforms,models
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch
    import torch.nn as nn
    import os
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from sklearn.metrics import classification_report
    import pandas as pd

    #use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    #setting tensorboard
    writer = SummaryWriter('../runs/vgg16_experiment')

    #load data
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    train_path = "../dataset/train"
    val_path = "../dataset/val"
    test_path = "../dataset/test"

    train_data = datasets.ImageFolder(train_path,transform = transform)
    val_data = datasets.ImageFolder(val_path,transform = transform)
    test_data = datasets.ImageFolder(test_path,transform = transform)

    train_loader = DataLoader(train_data,batch_size=16,shuffle=True,num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_data,batch_size=16,shuffle=False,num_workers=2,pin_memory=True)
    test_loader = DataLoader(test_data,batch_size=16,shuffle=False,num_workers=2,pin_memory=True)

    #vgg16

    model = models.vgg16(pretrained = True)
    model.classifier[6] = nn.Linear(4096,2)

    model = model.to(device)

    if os.path.exists('best_model_vgg16.pth'):
        print('find trained model,load')
        model.load_state_dict(torch.load('best_model_vgg16.pth'))
        model.eval()
        skip_training = True
    else:
        print('train')
        skip_training = False

    #loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-4)

    #train
    epochs = 10
    train_losses,val_losses = [],[]
    train_accuracies,val_accuracies = [],[]
    best_acc = 0.0

    if not skip_training:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct,total = 0,0

            for images,labels in train_loader:
                images,labels = images.to(device),labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                running_loss+=loss.item()
                k,preds = torch.max(outputs,1)
                total+=labels.size(0)
                correct+=(preds==labels).sum().item()

            train_loss = running_loss/len(train_loader)
            train_acc = correct/total

            model.eval()
            val_loss = 0.0
            correct,total = 0,0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = correct / total

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            # save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_model_vgg16.pth')
                print(f"\u2714\ufe0f bestmodel saved！（Val Acc: {best_acc:.4f}）")

            # SummaryWriter
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
        # save train data
        torch.save({
            'train_losses': torch.tensor(train_losses),
            'val_losses': torch.tensor(val_losses),
            'train_accuracies': torch.tensor(train_accuracies),
            'val_accuracies': torch.tensor(val_accuracies),
        }, 'training_logs_vgg16.pth')
        print("training data saved as 'training_logs_vgg16.pth'")

        writer.close()

    else:
        logs = torch.load('training_logs_vgg16.pth')
        train_losses = logs['train_losses'].tolist()
        val_losses = logs['val_losses'].tolist()
        train_accuracies = logs['train_accuracies'].tolist()
        val_accuracies = logs['val_accuracies'].tolist()
        print('skip training')


    #draw figure
    plt.figure(figsize=(12, 5))

    #Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    #test
    model.load_state_dict(torch.load('best_model_vgg16.pth'))
    model.eval()

    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

    # classification report
    # print report
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=['No Tumor', 'Tumor']))

    # save report as CSV

    report_dict = classification_report(all_labels, all_preds, target_names=['No Tumor', 'Tumor'], output_dict=True)
    df_report = pd.DataFrame(report_dict).T
    os.makedirs("results",exist_ok=True)
    df_report.to_csv("./results/classification_report.csv")
    print("Classification report saved to classification_report.csv")

    # Tumor recall and f1-score
    recall_tumor = report_dict['Tumor']['recall']
    f1_tumor = report_dict['Tumor']['f1-score']
    print(f"Tumor Recall: {recall_tumor:.4f}")
    print(f"Tumor F1-score: {f1_tumor:.4f}")

    # misclassified samples and error rate
    num_errors = sum(1 for i in range(len(all_preds)) if all_preds[i] != all_labels[i])
    error_rate = num_errors / len(all_preds)
    print(f"Number of misclassified samples: {num_errors}")
    print(f"Error rate: {error_rate:.2%}")

    #confusion mat
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
