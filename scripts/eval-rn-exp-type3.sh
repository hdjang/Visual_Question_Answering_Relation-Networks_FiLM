CUDA_VISIBLE_DEVICES=4 \
python main.py  --exp_id      rn-exp-type3   \
                --model       RN             \
                --phase       test           \
                --num_workers 40             \
                --epochs      40             \
                --batch_size  64             \
                \
                --multi_step  15,18,20       \
                --lr          0.0001         \
                --lr_max      0.01           \
                --lr_gamma    2              \
                \
                --cnn_chs     128,128,128,128  \
                --rn_g_chs    512,512          \
                --rn_f_chs    512,512          \
                \
                --checkpoint  './exp/rn-exp-type3/RN_epoch_35-93acc.pth' \
                #>./log/rn-exp-type3