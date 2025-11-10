from main import *
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda")

# 对训练集数据增强
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.RandomCrop(32, padding=4),  # 随机裁剪
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 数据集准备
train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=train_transform,
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=train_transform,
                                          download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 模型初始化
model = VisionNet()
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 学习率调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# 训练参数
total_train_step = 0 # 训练次数
total_test_step = 0 # 测试次数 
epoch = 50 # 训练轮数

# 训练过程可视化
writer = SummaryWriter("./logs_train")
cumulative_test_loss = 0
cumulative_accuracy = 0
num_test_batches = len(test_dataloader)

start_time = time.time()
best_accuracy = 0
best_epoch = 0

for i in range(epoch):
    print(f"\n第{i+1}轮训练：")

    # 训练步骤开始
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(f"训练次数：{total_train_step}, Loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 计算正确预测的样本数
        correct = (outputs.argmax(1) == targets).sum().item()
        total_train_correct += correct
        # 累加训练损失
        total_train_loss += loss.item()

    # 计算训练集上的平均Loss和正确率
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_correct / len(train_dataloader.dataset)

    writer.add_scalar("train_loss_avg", avg_train_loss, total_train_step)
    writer.add_scalar("train_accuracy_avg", avg_train_accuracy, total_train_step)

    # 更新学习率
    scheduler.step()

    # 测试
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            # 累加每个批次的Loss和正确率
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy +=accuracy

        # 增加测试步骤计数器
        total_test_step += 1
        # 计算测试集的平均Loss和正确率
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_accuracy = total_accuracy / len(test_dataloader.dataset)
        # 记录测试集的平均Loss和正确率到TensorBoard
        writer.add_scalar("test_loss_avg", avg_test_loss, total_test_step)
        writer.add_scalar("test_accuracy_avg", avg_accuracy, total_test_step)
        # 重置累计值，为下一个epoch做准备
        cumulative_test_loss = 0
        cumulative_accuracy = 0

    # 保存模型
    current_accracy = avg_accuracy
    if current_accracy > best_accuracy:
        best_accuracy = current_accracy
        best_epoch = i
        torch.save(model.state_dict(), f"best_model.pth")
    print("最佳模型已保存，准确率为{best_accuracy:.4f}")

writer.close()
