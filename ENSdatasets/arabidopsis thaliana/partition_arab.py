import copy

acc_neg = './SubsetTest/Acceptor_Test_Negative.txt'
acc_pos = './SubsetTest/Acceptor_Test_Positive.txt'
don_neg = './SubsetTest/Donor_Test_Negative.txt'
don_pos = './SubsetTest/Donor_Test_Positive.txt'

with open(acc_neg, 'r') as f:
    all_lines = f.readlines()
    chars = 0
    file_index = 1
    new_lines = []
    lines = []
    for seq in all_lines:
        chars += len(seq)-2
        if (chars - 200000) > -612:
            chars = 0
            with open(f'./SubsetSpliceRover/Acceptor_Test_Negative_{file_index}.fa', 'w') as g:
                for line in lines:
                    g.write(line)
                lines = copy.deepcopy(new_lines)
                file_index+=1
                g.close()
        lines.append(seq)
    f.close()
    

with open(acc_pos, 'r') as f:
    all_lines = f.readlines()
    chars = 0
    file_index = 1
    new_lines = []
    lines = []
    for seq in all_lines:
        chars += len(seq)-2
        if (chars - 200000) > -612:
            chars = 0
            with open(f'./SubsetSpliceRover/Acceptor_Test_Positive_{file_index}.fa', 'w') as g:
                for line in lines:
                    g.write(line)
                lines = copy.deepcopy(new_lines)
                file_index+=1
                g.close()
        lines.append(seq)
    f.close()


with open(don_pos, 'r') as f:
    all_lines = f.readlines()
    chars = 0
    file_index = 1
    new_lines = []
    lines = []
    for seq in all_lines:
        chars += len(seq)-2
        if (chars - 200000) > -612:
            chars = 0
            with open(f'./SubsetSpliceRover/Donor_Test_Positive_{file_index}.fa', 'w') as g:
                for line in lines:
                    g.write(line)
                lines = copy.deepcopy(new_lines)
                file_index+=1
                g.close()
        lines.append(seq)
    f.close()


with open(don_neg, 'r') as f:
    all_lines = f.readlines()
    chars = 0
    file_index = 1
    new_lines = []
    lines = []
    for seq in all_lines:
        chars += len(seq)-2
        if (chars - 200000) > -612:
            chars = 0
            with open(f'./SubsetSpliceRover/Donor_Test_Negative_{file_index}.fa', 'w') as g:
                for line in lines:
                    g.write(line)
                lines = copy.deepcopy(new_lines)
                file_index+=1
                g.close()
        lines.append(seq)
    f.close()
