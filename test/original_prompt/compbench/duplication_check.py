file = 'color_val.txt'
with open(file, 'r') as f:
    prompts = f.readlines()

prompts = [prompt.rstrip() for prompt in prompts]
print(prompts)

new_prompts = list(set(prompts))
print(len(prompts), len(new_prompts)) # 300, 295



# TODO:
read = {}
for i, prompt in enumerate(prompts):
    if prompt not in read:
        print(i, prompt)
        read[prompt] = True
    else:
        print("ALREAD READ!!")
        print(i, prompt)
    