# The training code of Encodec

### Note that, this part of code is based on Facebook's Encodec. We just provide the training process. The license is the same as Encodec.

### For Training
set the right path to statr/start.sh

run: `bash start.sh`

### For Finetune
If you want to finetune the model, you can use following instruct: 
`
python3 main3_ddp.py --BATCH_SIZE 16  --N_EPOCHS 300 \
        --save_dir path_to_save_log \
        --PATH  path_to_save_model \
        --train_data_path path_to_training_data \
        --valid_data_path path_to_val_data \
        --resume  --resume_path the_model_path
`

### For Inference
if you want to use our checkpoint. Run the following <br>
```bash
mkdir checkpoint
cd checkpoint
wget https://huggingface.co/Dongchao/AcademiCodec/resolve/main/encodec_24khz_240d.pth
bash test.sh    # set the root in test.sh, before runing it.
```