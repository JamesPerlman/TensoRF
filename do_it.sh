TENSORF=/c/Users/bizon/Developer/TensoRF
PROJECT=/g/statue

cd $PROJECT && \
python $TENSORF/train.py \
    --config $TENSORF/configs/your_own_data.txt \
    --ckpt G:\\statue\\log\\tensorf_xxx_VM\\tensorf_xxx_VM.th \
    --render_only 1 \
    --render_path 1 \
    --render_test 0
    
