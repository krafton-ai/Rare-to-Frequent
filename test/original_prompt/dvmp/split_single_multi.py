test_file = 'dvmp_test500.txt'

with open(test_file) as f:
    prompts = [line.rstrip() for i, line in enumerate(f)]

#print(prompts)
print(len(prompts))

single, multi = [], []
for prompt in prompts:
    words = [w for w in prompt.split(' ')]

    if 'and' not in words:
        single.append(prompt)
    else:
        multi.append(prompt)
print(len(single), len(multi)) # 132, 368


out_single = 'dvmp_single100.txt'
with open(out_single, 'w+') as f:
    for prompt in single[:100]:
        f.write(prompt+'\n')

out_multi = 'dvmp_multi100.txt'
with open(out_multi, 'w+') as f:
    for prompt in multi[:100]:
        f.write(prompt+'\n')