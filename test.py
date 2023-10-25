from predict import predict_image
import time

url='/mnt/d/final-project/images/char-4-epoch-1/test/'
images=['hbpi_2e3d1942-85fa-44f0-81c4-458524506a55.png','gxti_0486eea8-3711-4878-beb7-6723107e3896.png','dyap_c4c20779-5bc3-488d-812e-066959fec1f8.png','dtym_5fa179ae-00b3-4320-b023-ac7104a264c7.png','qrpt_5a5e7f60-6f02-4c99-8197-8f7ce214e558.png','raqp_85070ff8-01cb-4305-8e95-5338f0b1de38.png','rdhq_f883d7a6-1a28-4425-81b1-c36b9adce8d7.png','rdtp_7b613265-54b2-4195-9a39-c89bd5caef08.png','ripm_e7dfbf70-6079-457f-9ebe-8f29457cb908.png','rvbp_8354bd9d-fee7-4df3-b815-457b1055db08.png']

for img in images:
    print(predict_image(url+img))
