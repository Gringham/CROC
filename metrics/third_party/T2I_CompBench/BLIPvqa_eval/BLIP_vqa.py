import os
import torch
import json
import spacy

from tqdm import tqdm
from tqdm.auto import tqdm

from vmen.metrics.third_party.T2I_CompBench.BLIPvqa_eval.BLIP.train_vqa_func import VQA_main
from vmen.project_root import join_with_root

def Create_annotation_for_BLIP(prompts, outpath, np_index=None, dataset_name='default'):
    nlp = spacy.load("en_core_web_sm")

    annotations = []
    cnt=0

    #output annotation.json
    for path, prompt in prompts:
        image_dict={}
        image_dict['image'] = path
        image_dict['question_id']= cnt
        doc = nlp(prompt)
        
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.text not in ['top', 'the side', 'the left', 'the right']:  # todo remove some phrases
                noun_phrases.append(chunk.text)
        if(len(noun_phrases)>np_index):
            q_tmp = noun_phrases[np_index]
            image_dict['question']=f'{q_tmp}?'
        else:
            image_dict['question'] = ''
            

        image_dict['dataset']=dataset_name
        annotations.append(image_dict)
        cnt+=1

    with open(f'{outpath}/vqa_test.json', 'w') as f:
        f.write(json.dumps(annotations))

def main(image_prompt_dict, out_dir="bvqa_intermediate", np_num=8):
    #torch.cuda.set_device(1)
    answer = []
    sample_num = len(image_prompt_dict)
    reward = torch.zeros((sample_num, np_num)).to(device='cuda')

    order="_blip" #rename file
    for i in tqdm(range(np_num)):
        print(f"start VQA{i+1}/{np_num}!")
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}", exist_ok=True)
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}/VQA/", exist_ok=True)
        Create_annotation_for_BLIP(
            image_prompt_dict,
            f"{out_dir}/annotation{i + 1}{order}",
            np_index=i,
        )
        answer_tmp = VQA_main(f"{out_dir}/annotation{i + 1}{order}/",
                              f"{out_dir}/annotation{i + 1}{order}/VQA/")
        answer.append(answer_tmp)

        with open(f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
            r = json.load(file)
        with open(f"{out_dir}/annotation{i + 1}{order}/vqa_test.json", "r") as file:
            r_tmp = json.load(file)
        for k in range(len(r)):
            if(r_tmp[k]['question']!=''):
                reward[k][i] = float(r[k]["answer"])
            else:
                reward[k][i] = 1
        print(f"end VQA{i+1}/{np_num}!")
    reward_final = reward[:,0].clone() # Adding the copy here to add some clarity. Otherwise this will overwrite the original reward tensor.
    for i in range(1,np_num):
        reward_final *= reward[:,i]

    #output final json
    with open(f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
        r = json.load(file)
    reward_after=0
    for k in range(len(r)):
        r[k]["answer"] = '{:.4f}'.format(reward_final[k].item()) # Ugh, here they are setting it correct again, but pretty confusing.
        reward_after+=float(r[k]["answer"])
    os.makedirs(f"{out_dir}/annotation{order}", exist_ok=True)
    with open(f"{out_dir}/annotation{order}/vqa_result.json", "w") as file:
        json.dump(r, file)

    # calculate avg of BLIP-VQA as BLIP-VQA score
    #print("BLIP-VQA score:", reward_after/len(r),'!\n')
    #with open(f"{out_dir}/annotation{order}/blip_vqa_score.txt", "w") as file:
    #    file.write("BLIP-VQA score:"+str(reward_after/len(r)))

    return reward_final



if __name__ == "__main__":
    paths = ['cat1.bmp', 
             'cat2.bmp', 
             'chair.bmp', 
             'combined.bmp', 
             'tiger.bmp',
             'cat1.bmp', 
             'cat2.bmp', 
             'chair.bmp', 
             'combined.bmp', 
             'tiger.bmp']*1000
    
    paths = [join_with_root("metrics/test_images/" + p) for p in paths]
    
    captions = ["A cute cat",
                "Another cute cat",
                "A stylish chair",
                "A combined image of a cat, a tiger, and a chair",
                "A majestic tiger",
                "An ugly dog",
                "A rusty old car",
                "A dilapidated building",
                "An empty white room",
                "A boring piece of paper"]*1000
    
    print("OIOIOIO", paths)
    
    input = [(p, c) for p, c in zip(paths, captions)]
    main(input)