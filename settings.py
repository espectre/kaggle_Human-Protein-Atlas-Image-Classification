import os

config_linux={
    'n_gpu':"0,1,2,3,4,5",
    'path_to_train':"/mnt/home2/yhf/protein/train/",
    'extra_data':"/mnt/home2/yhf/protein/external/",
    'extra_csv':"/mnt/home2/yhf/protein/HPAv18RBGY_wodpl.csv",
    'train_csv':"/mnt/home2/yhf/protein/train.csv",
    'save_model':'../working/',
    'sample_submission':"/mnt/home2/yhf/protein/sample_submission.csv",
    'path_to_test':"/mnt/home2/yhf/protein/test/",
    'load_model_path':'./result3/b12_extra_pretrian10/swa_weight.ckpt',#load the model
    'batch_size':128,
    'save_transform_size':"/home/gujiang/protein/train_768/",
    'epochs':60,
    'SIZE':256,
    'num_workers':8,
    'resume':None,
    'start_epoch':0,
    'lr':1e-3,
    'weight_decay':None,
    'save_freq':1,
    'save_dir':"",
    'finetune_model':'./result2/b12_finetune_155/10_0.129381.ckpt',
    "swa_list":["train_40_acc_0.669351.ckpt","train_41_acc_0.667890.ckpt","train_42_acc_0.668100.ckpt",
                "train_43_acc_0.668216.ckpt","train_44_acc_0.669173.ckpt"],
    "model_average":[],
    "save_swa_weight":"None",
    "model_path_root":"./result3/b12_extra_pretrian10/",
    'Kfold':True
}


#train_5_acc_0.435667.ckpt


config=config_linux