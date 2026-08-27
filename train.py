from utils.regression_trainer_cosine_multibatch import RegTrainer
import argparse
import os
import torch
args = None

# 設定「執行訓練程式時可以從終端機傳入的參數」
def parse_args():
    # 建立參數解析器
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--model-name', default='vgg19_trans', help='the name of the model')
    parser.add_argument('--pretrain-data-dir', default=r'/media/mmslab-1080/37E8097A1DE5832D/Kuan-Lun/counting_data/mix_data/ShanghaiTech_A_Train_Val_Test',
                        help='pretrain data directory')
    parser.add_argument('--save-dir', default='model',
                        help='directory to save models.')
    parser.add_argument('--save-all', type=bool, default=False,
                        help='whether to save all best model')
    parser.add_argument('--pretrain-lr', type=float, default=5*1e-6,
                        help='learning rate for base-model pretraining')
    parser.add_argument('--pretrain-weight-decay', type=float, default=1e-5,
                        help='weight decay for base-model pretraining')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--pretrain-epochs', type=int, default=10,
                        help='number of base-model pretraining epochs')
    parser.add_argument('--pretrain-val-epoch', type=int, default=5,
                        help='validate the base model every N pretrain epochs')
    parser.add_argument('--pretrain-val-start', type=int, default=0,
                        help='first zero-based pretrain epoch eligible for validation')

    parser.add_argument('--train-data-dir', default=r'/media/mmslab-1080/37E8097A1DE5832D/Kuan-Lun/counting_data/mix_data/ShanghaiTech_A_Train_Val_Test',
                        help='LoRA/router training data directory')
    parser.add_argument('--lr', '--lora-lr', dest='lr', type=float, default=5*1e-6,
                        help='learning rate for LoRA/router training')
    parser.add_argument('--weight-decay', '--lora-weight-decay', dest='weight_decay',
                        type=float, default=1e-5,
                        help='weight decay for LoRA/router training')
    parser.add_argument('--lora-epochs', '--max-epoch', dest='lora_epochs',
                        type=int, default=10,
                        help='number of LoRA/router training epochs')    
    parser.add_argument('--lora-val-epoch', '--val-epoch', dest='lora_val_epoch',
                        type=int, default=5,
                        help='validate LoRA/router every N LoRA training epochs')
    parser.add_argument('--lora-val-start', '--val-start', dest='lora_val_start',
                        type=int, default=0,
                        help='first zero-based LoRA epoch eligible for validation')
    
    #gamma是用於輔助損失的路由器偏差更新的學習率，控制了路由器偏差更新的速度
    parser.add_argument('--router-bias-update-rate', type=float, default=1e-3,
                        help='gamma for auxiliary-loss-free router bias updates')   
    
    
    parser.add_argument('--batch-size', type=int, default=4,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=16,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)   # train.py 本身不負責寫完整訓練流程，而是交給 RegTrainer
    trainer.setup()
    trainer.train()
