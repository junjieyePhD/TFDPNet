export CUDA_VISIBLE_DEVICES=0


model_name=TFDPNet


for pred_len in 12 24 48
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/PEMS/ \
      --data_path PEMS08.npz \
      --model_id PEMS08_96_$pred_len'_'$alpha \
      --model $model_name \
      --data PEMS \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --use_norm 1 \
      --enc_in 170 \
      --des 'Exp' \
      --d_model 256 \
      --d_model_frequency 256 \
      --learning_rate 0.001 \
      --batch_size 16 \
      --train_epochs 10 \
      --dropout 0.1
done


for pred_len in 96
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/PEMS/ \
      --data_path PEMS08.npz \
      --model_id PEMS08_96_$pred_len \
      --model $model_name \
      --data PEMS \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 170 \
      --des 'Exp' \
      --d_model 256 \
      --learning_rate 0.001 \
      --batch_size 16 \
      --train_epochs 10 \
      --d_model_frequency 256 \
      --use_norm 0 \
      --dropout 0.1
done


