export CUDA_VISIBLE_DEVICES=0


model_name=TFDPNet


for pred_len in 12 24 48 96
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/PEMS/ \
      --data_path PEMS07.npz \
      --model_id PEMS07_96_$pred_len'_'$alpha \
      --model $model_name \
      --data PEMS \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --use_norm 0 \
      --enc_in  883 \
      --des 'Exp' \
      --d_model 512 \
      --learning_rate 0.001 \
      --batch_size 16 \
      --train_epochs 10 \
      --dropout 0.1
done




