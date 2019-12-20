EPOCH=3
GAUSSIAN=4 #0.

echo -e "\033[0;31m[*]NORMAL->NORMAL, GAUSSIAN: 0.${GAUSSIAN}\033[0;39m"
python keras_mnist.py --epochs ${EPOCH}
echo "[*]mv trainlog.csv ./train_logs/trainlog_gau${GAUSSIAN}_normal-normal_epoch${EPOCH}.csv"
mv trainlog.csv ./train_logs/trainlog_gau${GAUSSIAN}_normal-normal_epoch${EPOCH}.csv

echo -e "\033[0;31m[*]NORMAL->NOISE, GAUSSIAN: 0.${GAUSSIAN}\033[0;39m"
python keras_mnist.py --epochs ${EPOCH} --test_noise
echo "[*]mv trainlog.csv ./train_logs/trainlog_gau${GAUSSIAN}_normal-noise_epoch${EPOCH}.csv"
mv trainlog.csv ./train_logs/trainlog_gau${GAUSSIAN}_normal-noise_epoch${EPOCH}.csv

echo -e "\033[0;31m[*]NOISE->NORMAL, GAUSSIAN: 0.${GAUSSIAN}\033[0;39m"
python keras_mnist.py --epochs ${EPOCH} --train_noise
echo "[*]mv trainlog.csv ./train_logs/trainlog_gau${GAUSSIAN}_noise-normal_epoch${EPOCH}.csv"
mv trainlog.csv ./train_logs/trainlog_gau${GAUSSIAN}_noise-normal_epoch${EPOCH}.csv

echo -e "\033[0;31m[*]NOISE->NOISE, GAUSSIAN: 0.${GAUSSIAN}\033[0;39m"
python keras_mnist.py --epochs ${EPOCH} --train_noise --test_noise
echo "[*] mv trainlog.csv ./train_logs/trainlog_gau${GAUSSIAN}_noise-noise_epoch${EPOCH}.csv"
mv trainlog.csv ./train_logs/trainlog_gau${GAUSSIAN}_noise-noise_epoch${EPOCH}.csv

echo -e "\033[0;32mCOMPLETED!\033[0;39m"
