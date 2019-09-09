CUDA_VISIBLE_DEVICES=5 \
python main.py  --exp_id      rn-exp-type3-2   \
                --model       RN             \
                --phase       train          \
                --num_workers 40             \
                --batch_size  64             \
                --epochs      35             \
                \
                --multi_step  15,20          \
                --lr          0.0001         \
                --lr_max      0.01           \
                --lr_gamma    1              \
                \
                --cnn_chs     128,128,128,128 \
                --rn_g_chs    512,512         \
                --rn_f_chs    512,512         \
                \
                #--skip_train_eval             \
                #>./log/rn-exp-type3