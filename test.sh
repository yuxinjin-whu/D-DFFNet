
python test.py --pretrained '/data/jyx/dbd/D-DFFNet/checkpoint/Ablation_study/DFFNet_ablation/D-DFFNet.pth' --test_dataset 'Shi'
python test.py --pretrained '/data/jyx/dbd/D-DFFNet/checkpoint/Ablation_study/DFFNet_ablation/D-DFFNet.pth' --test_dataset 'CTCUG'
python test.py --pretrained '/data/jyx/dbd/D-DFFNet/checkpoint/Ablation_study/DFFNet_ablation/D-DFFNet.pth' --test_dataset 'DUT'
python test.py --pretrained '/data/jyx/dbd/D-DFFNet/checkpoint/Ablation_study/DFFNet_ablation/D-DFFNet.pth' --test_dataset 'EBD'

python test_iou.py --pretrained '/data/jyx/dbd/D-DFFNet/checkpoint/Ablation_study/DFFNet_ablation/D-DFFNet.pth' --test_dataset 'Shi'
python test_iou.py --pretrained '/data/jyx/dbd/D-DFFNet/checkpoint/Ablation_study/DFFNet_ablation/D-DFFNet.pth' --test_dataset 'CTCUG' 
python test_iou.py --pretrained '/data/jyx/dbd/D-DFFNet/checkpoint/Ablation_study/DFFNet_ablation/D-DFFNet.pth' --test_dataset 'DUT'
python test_iou.py --pretrained '/data/jyx/dbd/D-DFFNet/checkpoint/Ablation_study/DFFNet_ablation/D-DFFNet.pth' --test_dataset 'EBD' 