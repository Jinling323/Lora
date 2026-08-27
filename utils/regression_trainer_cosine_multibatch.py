import logging 
import os  
import sys  
import time 
from math import ceil  

# This machine's protobuf C++ extension requires a newer libstdc++; the pure
# Python implementation keeps TensorBoard usable without changing training.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np  
import torch  
from torch import optim  
from torch.utils.data import DataLoader  
from torch.utils.data.dataloader import default_collate  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.helper import AverageMeter, Save_Handle 
from utils.trainer import Trainer 

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) 
from datasets.crowd import Crowd  
from losses.bay_loss import Bay_Loss  
from losses.post_prob import Post_Prob 
from models import vgg_c_multibatch as vgg  


def train_collate(batch):  
    columns = list(zip(*batch))  
    images = torch.stack(columns[0], dim=0)  
    points = columns[1]  
    targets = columns[2]  
    st_sizes = torch.FloatTensor(columns[3])  
    return images, points, targets, st_sizes  


class RegTrainer(Trainer):  
    def setup(self): 
        args = self.args  
        if not torch.cuda.is_available(): 
            raise RuntimeError("gpu is not available") 

        self.device = torch.device("cuda")  
        self.device_count = torch.cuda.device_count() 
        assert self.device_count == 1, "only single-GPU training is supported" 
        logging.info("using %s gpu", self.device_count)  

        self.datasets = self._make_datasets(args.train_data_dir)
        self.dataloaders = self._make_dataloaders(self.datasets) 
        pretrain_data_dir = args.pretrain_data_dir or args.train_data_dir  
        same_data = os.path.abspath(pretrain_data_dir) == os.path.abspath(args.train_data_dir)  
        self.pretrain_datasets = self.datasets if same_data else self._make_datasets(pretrain_data_dir) 
        self.pretrain_dataloaders = self.dataloaders if same_data else self._make_dataloaders(self.pretrain_datasets)  
        logging.info("pretrain data directory: %s", pretrain_data_dir)  

        self.model = vgg.vgg19_trans(  
            num_experts=4,  
            top_k=2,  
            lora_rank=4,  
            lora_alpha=4.0,  
            freeze_vgg=False,  
        )
        self.model.to(self.device)  

        base_parameters = self.model.base_parameters()  
        lora_parameters = self.model.lora_parameters() 

        # 建立 base pretrain 專用 optimizer。
        pretrain_optimizer = optim.Adam(  
            base_parameters,  # 此 optimizer 永遠只管理 base parameters。
            lr=args.pretrain_lr,  # 使用 base 階段 learning rate。
            weight_decay=args.pretrain_weight_decay,  # 使用 base 階段 weight decay。
        )
        # 建立 LoRA/router 專用 optimizer。
        lora_optimizer = optim.Adam(  
            lora_parameters,  # 此 optimizer 永遠只管理 LoRA/router parameters。
            lr=args.lr,  # args.lr 對應 --lora-lr。
            weight_decay=args.weight_decay,  
        )
        # 記錄兩個參數群組大小，方便確認凍結範圍。
        logging.info(  
            "base parameters: {:,}; LoRA/router parameters: {:,}".format(
                sum(parameter.numel() for parameter in base_parameters),  # 計算 base 參數總數。
                sum(parameter.numel() for parameter in lora_parameters),  # 計算 LoRA/router 參數總數。
            )
        )

        # Base 權重放在實驗目錄的 pretrain 子目錄。
        self.pretrain_dir = os.path.join(self.save_dir, "pretrain")  
        os.makedirs(self.pretrain_dir, exist_ok=True) 
        pretrain_saves = Save_Handle(args.max_model_num)  # 控制 base 一般 checkpoint 保留數量。
        lora_saves = Save_Handle(args.max_model_num)  # 控制 LoRA 一般 checkpoint 保留數量。

        self.stages = {  # 把兩階段的差異集中管理，避免重複兩套訓練程式。
            "base": {  # Base model pretrain 設定。
                "label": "Base pretrain", 
                "epochs": args.pretrain_epochs, 
                "val_interval": args.pretrain_val_epoch, 
                "val_start": args.pretrain_val_start,  
                "loaders": self.pretrain_dataloaders,  
                "optimizer": pretrain_optimizer,  
                "save_dir": self.pretrain_dir,  # Base checkpoint 儲存資料夾。
                "save_handle": pretrain_saves,  
            },
            "lora": {  # LoRA experts/router fine-tuning 設定。
                "label": "LoRA/router",  
                "epochs": args.lora_epochs,  
                "val_interval": args.lora_val_epoch,  
                "val_start": args.lora_val_start, 
                "loaders": self.dataloaders,  
                "optimizer": lora_optimizer,  
                "save_dir": self.save_dir,  # LoRA checkpoint 儲存在實驗主目錄。
                "save_handle": lora_saves,  
            },
        }
        for stage in self.stages.values():  #檢查args.lora_val_epoch是否會設為小於等於0
            if stage["val_interval"] <= 0:  
                raise ValueError("validation interval must be greater than zero")  
        if args.router_bias_update_rate < 0: #檢查args.router_bias_update_rate是否小於0
            raise ValueError("router bias update rate must be non-negative")

        self.start_epochs = {"base": 0, "lora": 0}  # 預設兩階段都從 epoch 0 開始。
        self.best = {  # 分別追蹤 base 與 LoRA 的最佳 validation 結果。
            "base": {"mae": np.inf, "mse": np.inf, "count": 0},  
            "lora": {"mae": np.inf, "mse": np.inf, "count": 0},  
        }
        self.best_base_model_path = None  # 記錄本次執行實際保存的最佳 base 路徑。
        self.load_best_base_before_lora = True  # 完整流程預設在 LoRA 前載入最佳 base。
        self.save_all = args.save_all  # 預設為false，僅保留一個最佳模型檔案。

        self.post_prob = Post_Prob(  
            args.sigma,  
            args.crop_size,  
            args.downsample_ratio,  
            args.background_ratio, 
            args.use_background,  
            self.device, 
        )
        # 建立 Bayesian counting loss
        self.criterion = Bay_Loss(args.use_background, self.device)  
        if args.resume:  # 使用者有提供 checkpoint 時才執行
            self._load_resume(args.resume)  # 載入模型、optimizer 與起始 epoch。

        # TensorBoard logs are kept with the checkpoints and train.log for
        # this run. Base and LoRA use separate tags, so their epoch counters
        # can both start at zero without mixing the curves.
        self.tensorboard_dir = os.path.join(self.save_dir, "tensorboard")
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        logging.info("TensorBoard log directory: %s", self.tensorboard_dir)

    # 從指定根目錄建立 train 與 val datasets
    def _make_datasets(self, data_dir):  
        args = self.args  
        return {  
            split: Crowd(  
                os.path.join(data_dir, split),  
                args.crop_size,  
                args.downsample_ratio,  
                args.is_gray,  
                split,  
            )
            for split in ("train", "val")  
        }

    # 將 train/val datasets 包裝成 dataloaders
    def _make_dataloaders(self, datasets):  
        args = self.args  
        return { 
            split: DataLoader(  
                datasets[split],  
                collate_fn=train_collate if split == "train" else default_collate,  
                batch_size=args.batch_size if split == "train" else 1, 
                shuffle=split == "train",  
                num_workers=args.num_workers * self.device_count,  
                pin_memory=split == "train",  
            )
            for split in ("train", "val")  # 同時建立 train 與 val loaders。
        }

    def train(self):  # 依序執行 base pretrain 與 LoRA/router fine-tuning。
        try:
            self._run_stage("base")  # 先訓練 base pretrain。
            if self.stages["base"]["epochs"] > 0 and self.load_best_base_before_lora:  # 完整流程才需要切到最佳 base。
                self._load_best_base_before_lora()  # 以 validation 最佳 base 作為 LoRA 起點。
            self.stages["base"]["optimizer"].state.clear()
            torch.cuda.empty_cache()  # 刪除 Base Adam optimizer 的 moments，釋放 GPU 記憶體
            self._run_stage("lora")  # 接著執行或恢復 LoRA/router 訓練。
        finally:
            # Also close the event file when training exits because of an error.
            self.writer.close()

    def _run_stage(self, stage_name):  # 使用同一個流程執行 base 或 LoRA 階段。
        stage = self.stages[stage_name]  # 取得此階段集中管理的設定。
        for epoch in range(self.start_epochs[stage_name], stage["epochs"]):  # 從 resume epoch 執行到設定總數。
            self.epoch = epoch 
            logging.info("-----%s epoch %s/%s-----", stage["label"], epoch, stage["epochs"] - 1)  
            self._train_one_epoch(stage_name, stage)  
            should_validate = (epoch + 1) % stage["val_interval"] == 0  # 判斷是否已完成指定輪數。
            if should_validate and epoch >= stage["val_start"]:  # 同時滿足間隔與起始 epoch 才驗證。
                self._validate(stage_name, stage["loaders"]["val"])  # 執行此階段對應的 validation。

    # 執行 base 或 LoRA 的一個 training epoch
    def _train_one_epoch(self, stage_name, stage):  
        loss_meter = AverageMeter() 
        mae_meter = AverageMeter() 
        mse_meter = AverageMeter() 
        start_time = time.time()  
        optimizer = stage["optimizer"]  

        self.model.set_training_stage(stage_name)  # 自動設定 base/LoRA 的 requires_grad 與 forward 開關。
        self.model.train()  
        self.model.zero_grad(set_to_none=True)  # 清除上一階段可能殘留的所有 gradients。
        train_loader = stage["loaders"]["train"]
        progress = tqdm(
            train_loader,
            desc="{} epoch {}/{} train".format(stage["label"], self.epoch + 1, stage["epochs"]),
            unit="batch",
            dynamic_ncols=True,
        )
        for inputs, points, targets, st_sizes in progress:  
            optimizer.zero_grad()  # 清除此 optimizer 上一個 batch 的 gradients。
            if stage_name == "lora": # LoRA/router 階段才需要清除路由統計數據
                self.model.reset_router_loads()
            inputs = inputs.to(self.device)  
            st_sizes = st_sizes.to(self.device) 
            ground_truth = np.asarray([len(point) for point in points], dtype=np.float32) 
            points = [point.to(self.device) for point in points]  
            targets = [target.to(self.device) for target in targets]  

            outputs, features = self.model(inputs)  
            probabilities = self.post_prob(points, st_sizes)  
            bayesian_loss = self.criterion(probabilities, targets, outputs)  
            loss = bayesian_loss + self._consistency_loss(features, outputs)  
            loss.backward() 
            optimizer.step()  
            router_maxvio = None
            if stage_name == "lora": # LoRA/router 階段才需要更新路由器的 bias
                #router_maxvio是計算max violation（觀察用，不參與訓練）
                router_maxvio = self.model.update_router_biases( 
                    self.args.router_bias_update_rate 
                ) 

            batch_size = inputs.size(0)  
            predicted = outputs.reshape(batch_size, -1).sum(dim=1).detach().cpu().numpy() 
            residual = predicted - ground_truth  
            loss_meter.update(loss.item(), batch_size) 
            mse_meter.update(np.mean(residual ** 2), batch_size) 
            mae_meter.update(np.mean(np.abs(residual)), batch_size)  
            postfix = dict(
                loss="{:.2f}".format(loss_meter.get_avg()),
                mse="{:.2f}".format(np.sqrt(mse_meter.get_avg())),
                mae="{:.2f}".format(mae_meter.get_avg()),
            )
            if router_maxvio is not None:
                #新增maxvio到postfix，顯示最大違規比例
                postfix["maxvio"] = "{:.3f}".format(router_maxvio) 
            progress.set_postfix(postfix)

        logging.info(  
            "Epoch %s %s Train, Loss: %.2f, MSE: %.2f MAE: %.2f, Cost %.1f sec",
            self.epoch,  
            stage_name,  
            loss_meter.get_avg(),  
            np.sqrt(mse_meter.get_avg()),  
            mae_meter.get_avg(),  
            time.time() - start_time,  
        )
        tensorboard_prefix = "Base" if stage_name == "base" else "LoRA"
        self.writer.add_scalar(
            "{}/Train_Loss".format(tensorboard_prefix),
            loss_meter.get_avg(),
            self.epoch,
        )
        self.writer.add_scalar(
            "{}/Train_MAE".format(tensorboard_prefix),
            mae_meter.get_avg(),
            self.epoch,
        )
        self.writer.add_scalar(
            "{}/Train_RMSE".format(tensorboard_prefix),
            np.sqrt(mse_meter.get_avg()),
            self.epoch,
        )
        self.writer.flush()
        self._save_checkpoint(stage_name, stage)  # 每個 epoch 結束保存可續訓 checkpoint。

    @staticmethod
    # 計算原本程式使用的 cosine consistency loss
    def _consistency_loss(features, outputs):  
        total = outputs.new_zeros(())  
        for layer_features in features: 
            for feature in layer_features:  
                mean_feature = feature.mean(dim=0)  
                mean_norm = torch.sum(mean_feature ** 2) ** 0.5  
                feature_norm = torch.sum(feature ** 2, dim=1) ** 0.5  
                cosine_distance = 1 - torch.sum(feature * mean_feature, dim=1) / (mean_norm * feature_norm + 1e-5)  
                total = total + cosine_distance.sum()  
        return total  # 回傳 scalar loss。

    # 驗證 base-only 或 frozen-base + LoRA/router
    def _validate(self, stage_name, dataloader):  
        start_time = time.time()  # 記錄 validation 開始時間。
        self.model.enable_lora(stage_name == "lora")  # Base 驗證關閉 LoRA；LoRA 驗證開啟 LoRA/router。
        self.model.eval()  # 關閉 dropout 等 train的行為。
        residuals = []  # 收集每張 validation image 的 counting error。
        stage_label = self.stages[stage_name]["label"]

        #進度條
        progress = tqdm(
            dataloader,
            desc="{} epoch {}/{} val".format(
                stage_label,
                self.epoch + 1,
                self.stages[stage_name]["epochs"],
            ),
            unit="image",
            dynamic_ncols=True,
        )
        with torch.no_grad():  
            for inputs, count, _ in progress:  
                inputs = inputs.to(self.device)  
                assert inputs.size(0) == 1, "validation batch size must equal 1"  
                predicted_count = self._predict_count(inputs)  
                residuals.append(count[0].item() - predicted_count)  # 保存 ground truth 減 prediction。
                current_residuals = np.asarray(residuals)
                progress.set_postfix(
                    mse="{:.2f}".format(np.sqrt(np.mean(current_residuals ** 2))),
                    mae="{:.2f}".format(np.mean(np.abs(current_residuals))),
                )

        residuals = np.asarray(residuals)  
        mse = np.sqrt(np.mean(residuals ** 2))  # 計算 validation RMSE。
        mae = np.mean(np.abs(residuals))  # 計算 validation MAE。
        # 將 validation 指標寫入 train.log
        logging.info(  
            "Epoch %s %s Val, MSE: %.2f MAE: %.2f, Cost %.1f sec",
            self.epoch,  # 目前 zero-based epoch。
            stage_name,  # base 或 lora。
            mse,  # Validation RMSE。
            mae,  # Validation MAE。
            time.time() - start_time,  # Validation 花費秒數。
        )
        tensorboard_prefix = "Base" if stage_name == "base" else "LoRA"
        self.writer.add_scalar(
            "{}/Val_MAE".format(tensorboard_prefix), mae, self.epoch
        )
        self.writer.add_scalar(
            "{}/Val_RMSE".format(tensorboard_prefix), mse, self.epoch
        )
        self.writer.flush()
        # 指標刷新時保存此階段最佳模型。
        self._save_best(stage_name, mse, mae)  
        return mse, mae  

    # 將一張 validation image 轉成預測人數
    def _predict_count(self, inputs):  
        _, _, height, width = inputs.shape  # 取得影像高與寬
        if height < 3584 and width < 3584:  # 一般尺寸可以直接 forward
            return self.model(inputs)[0].sum().item()  # Density map 全部加總就是人數

        h_parts = int(ceil(float(height) / 3584))  # 計算垂直切割數量
        w_parts = int(ceil(float(width) / 3584))  # 計算水平切割數量
        h_step = height // h_parts  # 計算每個垂直區塊基本高度
        w_step = width // w_parts  # 計算每個水平區塊基本寬度
        predicted_count = 0.0  # 初始化所有區塊的人數總和
        for row in range(h_parts):  # 逐列處理影像區塊
            for column in range(w_parts):  # 逐欄處理影像區塊
                h_start = row * h_step  # 計算區塊上界
                h_end = (row + 1) * h_step if row < h_parts - 1 else height  
                w_start = column * w_step  # 計算區塊左界
                w_end = (column + 1) * w_step if column < w_parts - 1 else width  # 最後一欄包含剩餘像素
                tile = inputs[:, :, h_start:h_end, w_start:w_end]  # 從完整影像取出目前區塊
                predicted_count += self.model(tile)[0].sum().item()  # 累加此區塊 density map 的人數
        return predicted_count  # 回傳所有區塊的預測人數總和

    # 保存可精確續訓的 epoch checkpoint
    def _save_checkpoint(self, stage_name, stage):  
        path = os.path.join(stage["save_dir"], "{}_{}_ckpt.tar".format(stage_name, self.epoch))  # 組合 base/lora 檔名。
        torch.save(  # 將 epoch、stage、optimizer 與完整模型一起保存。
            {
                "epoch": self.epoch,  # 記錄目前完成的 zero-based epoch。
                "stage": stage_name,  # 記錄 checkpoint 屬於 base 或 lora。
                "optimizer_state_dict": stage["optimizer"].state_dict(),  # 保存對應 Adam state 供 resume。
                "model_state_dict": self.model.state_dict(),  # 保存完整模型權重。
            },
            path,  # 指定 checkpoint 寫入位置。
        )
        stage["save_handle"].append(path)  # 超過 max_model_num 時刪除最舊的一般 checkpoint。

    # 依 validation score 保存最佳 base 或 LoRA 模型。
    def _save_best(self, stage_name, mse, mae):  
        best = self.best[stage_name]  
        new_score = 2.0 * mse + mae  
        old_score = 2.0 * best["mse"] + best["mae"]  
        if new_score >= old_score:  
            return  # 保留原本最佳模型。

        best["mse"], best["mae"] = mse, mae  # 更新此階段最佳指標。
        output_dir = self.pretrain_dir if stage_name == "base" else self.save_dir  # Base 與 LoRA 使用不同資料夾。
        name = "best_base_model" if stage_name == "base" else "best_model"  # 選擇清楚的最佳模型檔名。
        if self.save_all:  #如果save_all為True，則保留每次刷新紀錄的最佳模型，檔名會加上編號。
            filename = "{}_{}.pth".format(name, best["count"])  # 例如 best_base_model_0.pth。
            best["count"] += 1  # 下一次刷新紀錄使用下一個編號。
        else:  # 預設只保留一個最佳模型。
            filename = "{}.pth".format(name)  # 每次刷新時覆寫相同最佳模型檔。
        path = os.path.join(output_dir, filename)  
        torch.save(self.model.state_dict(), path)  # 最佳模型只保存完整 model state。
        logging.info("saved best %s model: MSE %.2f MAE %.2f at %s", stage_name, mse, mae, path)  # 記錄保存結果。
        if stage_name == "base":  # 只有 base 最佳模型會成為下一階段初始化。
            self.best_base_model_path = path  # 記住本次執行剛保存的最佳 base 路徑。

    # 從 .pth 載入權重或從 .tar 精確恢復訓練。
    def _load_resume(self, resume_path):  
        suffix = resume_path.rsplit(".", 1)[-1].lower()  
        if suffix == "pth":  # .pth 只有模型權重，視為 LoRA 階段的初始化。
            self.model.load_state_dict(torch.load(resume_path, map_location=self.device))  # 載入完整 model state。
            self.start_epochs["base"] = self.stages["base"]["epochs"]  # 將 base 標記為已完成以跳過 pretrain。
            self.load_best_base_before_lora = False  # 使用指定權重，不再搜尋其他最佳 base。
            return  
        if suffix != "tar":  # 目前只支援 .pth 與 .tar。
            raise ValueError("--resume must point to a .pth or .tar file") 

        checkpoint = torch.load(resume_path, map_location=self.device)  # 讀取包含 stage/epoch/optimizer 的 checkpoint。
        self.model.load_state_dict(checkpoint["model_state_dict"])  
        stage_name = checkpoint.get("stage", "lora")  # 舊 checkpoint 沒有 stage 時沿用舊行為視為 LoRA。
        if stage_name not in self.stages:  # 避免未知 stage 使用錯誤 optimizer。
            raise ValueError("unknown checkpoint stage: {}".format(stage_name))  
        self._restore_optimizer(self.stages[stage_name]["optimizer"], checkpoint, stage_name)  # 恢復正確 stage 的 Adam state。
        self.start_epochs[stage_name] = checkpoint["epoch"] + 1  # 從 checkpoint 下一個 epoch 繼續。
        if stage_name == "lora":  # LoRA checkpoint 代表 base 階段已完成。
            self.start_epochs["base"] = self.stages["base"]["epochs"]  # 跳過 base pretrain。
            self.load_best_base_before_lora = False  # 不讓 base-only 權重覆蓋已訓練的 LoRA。

    # 嘗試恢復 base or lora 專用 optimizer。
    @staticmethod
    def _restore_optimizer(optimizer, checkpoint, stage_name):  
        try:  # 新舊 checkpoint optimizer 群組可能不同，因此需要容錯。
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 恢復 Adam moments 與 step。
        except (KeyError, ValueError) as error:  # 舊版混合 optimizer 無法對應新分組時進入此處。
            logging.warning("could not restore %s optimizer; starting fresh: %s", stage_name, error)  # 保留模型但重建 optimizer state。

    # 在完整流程中以 validation 最佳 base 初始化 LoRA。
    def _load_best_base_before_lora(self):  
        candidates = []  # 依優先順序收集可能的最佳 base 路徑。
        if self.best_base_model_path:  # 已有最佳 base 路徑時優先使用。
            candidates.append(self.best_base_model_path)  
        candidates.append(os.path.join(self.pretrain_dir, "best_base_model.pth"))  # 加入預設最佳檔名。
        candidates.extend(self._numbered_base_checkpoints(self.pretrain_dir))  # 支援 save_all 的編號檔名。
        if self.args.resume:  # 從 base .tar 恢復到新實驗目錄時也搜尋原 checkpoint 旁邊。
            resume_dir = os.path.dirname(os.path.abspath(self.args.resume))  # 取得 resume checkpoint 所在資料夾。
            candidates.append(os.path.join(resume_dir, "best_base_model.pth"))  # 搜尋原資料夾的預設最佳檔。
            candidates.extend(self._numbered_base_checkpoints(resume_dir))  # 搜尋原資料夾的 save_all 檔案。

        best_path = next((path for path in candidates if os.path.isfile(path)), None)  # 選擇第一個實際存在的候選檔。
        if best_path is None:  # 沒有做過 validation 或檔案被移除時可能找不到。
            logging.warning("best base checkpoint not found; using current base weights")  # 提醒改用目前最後權重。
            return  # 保持目前模型不變。
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))  # 載入 validation 最佳 base model。
        logging.info("loaded best base checkpoint for LoRA/router: %s", best_path)  # 記錄 LoRA 實際起始權重。

    # 尋找 save_all 產生的 best_base_model_N.pth。
    @staticmethod
    def _numbered_base_checkpoints(directory):  
        if not os.path.isdir(directory):  # 不存在的資料夾沒有候選 checkpoint。
            return []  # 回傳空 list。
        paths = [  # 收集所有符合命名規則的檔案。
            os.path.join(directory, filename)  
            for filename in os.listdir(directory)  # 逐一查看資料夾內容。
            if filename.startswith("best_base_model_") and filename.endswith(".pth")  # 只保留編號最佳 base。
        ]
        return sorted(paths, key=os.path.getmtime, reverse=True)  # 最新保存的最佳模型排在最前面。
