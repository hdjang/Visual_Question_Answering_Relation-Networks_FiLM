CUDA_VISIBLE_DEVICES=0 \
python main.py  --exp_id      RNs            \
                --model       RN             \
                --phase       train          \
                --num_workers 40             \
                --batch_size  64             \
                --epochs      35             \
                \
                --multi_step  15,20          \
                --lr          0.0001         \
                --lr_max      0.01           \
                --lr_gamma    2              \
                \
                --cnn_chs     128,128,128,128 \
                --rn_g_chs    512,512         \
                --rn_f_chs    512,512         \
