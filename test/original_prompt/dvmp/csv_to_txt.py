test_file = 'dvmp_test500.csv'

with open(test_file) as f:
    prompts = [line.split(",")[0].rstrip() for i, line in enumerate(f) if i!=0]

print(prompts)
print(len(prompts))

out_file = 'dvmp_test500.txt'

with open(out_file, 'w+') as f:
    for prompt in prompts:
        f.write(prompt+'\n')