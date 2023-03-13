import os

os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_dot_product")
os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_dot_product_times")
os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_dot_product_act_weighting --include_times=True")
os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_dot_product_act_weighting --include_times=False")
os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_w2vec --concat_method=single_sentence --include_times=True")
os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_w2vec --concat_method=full_sentence --include_times=True")
os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_w2vec --concat_method=weighting --include_times=True")
os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_w2vec --concat_method=single_sentence --include_times=False")
os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_w2vec --concat_method=full_sentence --include_times=False")
os.system("python pipeline_param.py --file bpi2017.xes --update_times_gen --emb_method=emb_w2vec --concat_method=weighting --include_times=False")