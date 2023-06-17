
# This is evaluation of D-DFFNet training on CUHK-TR-1 dataset, if you plan to evaluate other model, please change the pretrained path.

python test.py --pretrained './checkpoint/D-DFFNet.pth' --test_dataset 'Shi'
python test.py --pretrained './checkpoint/D-DFFNet.pth' --test_dataset 'CTCUG'
python test.py --pretrained './checkpoint/D-DFFNet.pth' --test_dataset 'DUT'
python test.py --pretrained './checkpoint/D-DFFNet.pth' --test_dataset 'EBD'

python test_iou.py --pretrained './checkpoint/D-DFFNet.pth' --test_dataset 'Shi'
python test_iou.py --pretrained './checkpoint/D-DFFNet.pth' --test_dataset 'CTCUG' 
python test_iou.py --pretrained './checkpoint/D-DFFNet.pth' --test_dataset 'DUT'
python test_iou.py --pretrained './checkpoint/D-DFFNet.pth' --test_dataset 'EBD' 
