# 本文件用于训练DeepCFD模型
import pickle
import json
from paddle.distributed import fleet
from utils.train_functions import *
from utils.functions import *
from model.UNetEx import UNetEx
import configparser

if __name__ == "__main__":
    fleet.init(is_collective=True)
    config = configparser.ConfigParser()
    config.read("./config/config.ini")

    # 加载数据集并处理
    x = pickle.load(open(os.path.join(config["path"]["data_path"], "dataLDFLPF.pkl"), "rb"))
    y = pickle.load(open(os.path.join(config["path"]["data_path"], "dataRad.pkl"), "rb"))
    x = paddle.to_tensor(x, dtype="float32")
    y = paddle.to_tensor(y, dtype="float32")
    y_trans = paddle.transpose(y, perm=[0, 2, 3, 1])
    channels_weights = paddle.reshape(
        paddle.sqrt(paddle.mean(paddle.transpose(y, perm=[0, 2, 3, 1]).reshape((808 * 192 * 192, 1)) ** 2, axis=0)),
        shape=[-1, 1, 1, 1]) #channels_weights 一个数，计算：把所有样本的所有数据平方后求平均，再开方，对于特定的数据集来说是定值

    # 创建保存文件夹
    simulation_directory = config["path"]["save_path"]
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)

    # 按7：3的比例分割数据集，7为训练集，3为测试集
    train_data, test_data = split_tensors(x, y, ratio=float(config["hyperparameter"]["train_test_ratio"]))

    train_dataset, test_dataset = paddle.io.TensorDataset([train_data[0], train_data[1]]), \
                                  paddle.io.TensorDataset([test_data[0], test_data[1]])
    test_x, test_y = test_dataset[:]

    # 设定种子，便于复现
    paddle.seed(999)
    # 设置训练epochs和batch_size
    epochs = int(config["hyperparameter"]["epochs"])
    batch_size = int(config["hyperparameter"]["batch_size"])
    # 设置学习率
    lr = float(config["hyperparameter"]["learning_rate"])
    # 设置卷积核大小
    kernel_size = int(config["net_parameter"]["kernel_size"])
    # 设置卷积层channel数目
    filters = [int(i) for i in config["net_parameter"]["filters"].split(",")]
    # 设置batch_norm和weight_norm
    bn = bool(int(config["net_parameter"]["batch_norm"]))
    wn = bool(int(config["net_parameter"]["weight_norm"]))
    # 构建模型
    model = UNetEx(2, 1, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    modelName = "DeepCFD_3523.pdparams"
    #model.set_state_dict(paddle.load(os.path.join(config["path"]["save_path"], config["path"]["model_name"])))
    #model.set_state_dict(paddle.load(os.path.join(config["path"]["save_path"], modelName)))
    model = fleet.distributed_model(model)
    # 定义优化器
    optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(),
                                       weight_decay=float(config["hyperparameter"]["weight_decay"]))
    optimizer = fleet.distributed_optimizer(optimizer)

    # 设置记录列表
    config = {} # 此处的config不同于上面的。此处往后用于记录结果
    train_loss_curve = []
    test_loss_curve = []
    train_mse_curve = []
    test_mse_curve = []
    
    


    # 用于后续训练过程的记录
    def after_epoch(scope):
        train_loss_curve.append(scope["train_loss"])
        test_loss_curve.append(scope["val_loss"])
        train_mse_curve.append(scope["train_metrics"]["mse"])
        test_mse_curve.append(scope["val_metrics"]["mse"])
        #train_ux_curve.append(scope["train_metrics"]["ux"])
        #test_ux_curve.append(scope["val_metrics"]["ux"])
        #train_uy_curve.append(scope["train_metrics"]["uy"])
        #test_uy_curve.append(scope["val_metrics"]["uy"])
        #train_p_curve.append(scope["train_metrics"]["p"])
        #test_p_curve.append(scope["val_metrics"]["p"])


    # 损失函数
    # def loss_func(model, batch):
    #     x, y = batch
    #     output = model(x)
    #     lossRad = paddle.abs((output[:, 0, :, :] - y[:, 0, :, :])).reshape(
    #         (output.shape[0], 1, output.shape[2], output.shape[3]))
    #     loss = lossRad / channels_weights
    #     return paddle.sum(loss), output

    def loss_func(model, batch):
        x, y = batch
        output = model(x)
        lossRad = paddle.abs((output[:, 0, :, :] - y[:, 0, :, :]) / y[:, 0, :, :]).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        # lossRad = paddle.abs(output.cpu() - y.cpu()) / y.cpu()
        loss = paddle.mean(lossRad) * 0.9 + paddle.std(lossRad) * 0.1
        return loss, output


    # 训练模型，加入除loss以外的1个指标：Total MSE
    DeepCFD, train_metrics, train_loss, test_metrics, test_loss = train_model(simulation_directory, model, loss_func,
                train_dataset, test_dataset, optimizer,
                epochs=epochs, batch_size=batch_size,
                m_mse_name="Total MSE",   # **kwargs，这边的参数数量不定，可以增加删除
                m_mse_on_batch=lambda scope: float(
                    paddle.sum((scope["output"] -
                                scope["batch"][1]) ** 2)),
                m_mse_on_epoch=lambda scope: sum(
                    scope["list"]) / len(
                    scope["dataset"]), 
                patience=25,
                after_epoch=after_epoch
                )

    # 用于记录训练过程中的各项指标并保存
    metrics = {}
    metrics["train_metrics"] = train_metrics
    metrics["train_loss"] = train_loss
    metrics["test_metrics"] = test_metrics
    metrics["test_loss"] = test_loss
    
    curves = {}
    curves["train_loss_curve"] = train_loss_curve
    curves["test_loss_curve"] = test_loss_curve
    curves["train_mse_curve"] = train_mse_curve
    curves["test_mse_curve"] = test_mse_curve
    
    # 将上述两者metrics和curves记录在.json中
    config["metrics"] = metrics
    config["curves"] = curves

    # 保存各项训练指标
    with open(os.path.join(simulation_directory, "results.json"), "w") as file:        
        json.dump(config, file)
        
