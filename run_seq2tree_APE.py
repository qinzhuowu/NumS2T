# coding: utf-8
from __future__ import division 
from pre_data import *
from masked_cross_entropy import *
from expressions_transfer import *
from models import *

import time
import torch.optim
from torch.optim import lr_scheduler

USE_CUDA = torch.cuda.is_available()

batch_size = 64
embedding_size = 300
hidden_size = 512+128
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 3
n_layers = 2


dataset="APE"
# 加载数据集
if dataset=="APE":
    train_data = load_data('data/ape/train.ape.json',1)
    valid_data = load_data('data/ape/valid.ape.json',1)
    test_data = load_data('data/ape/test.ape.json',1)
    n_epochs = 50
else:
    # 加载数据集
    train_data = load_Math23K_data('data/Math_23K.json')
    print(len(train_data))

    fold_size1 = int(len(train_data) * 0.8)
    fold_size2 = int(len(train_data) * 0.9)
    valid_data = train_data[fold_size1:fold_size2]
    test_data = train_data[fold_size2:]
    train_data = train_data[:fold_size1]
    n_epochs = 80

number_char_vocab=['UNK','0', '1', '2', '3', '4','5','6','7','8','9','.', '(', '/', ')', '%']
number_char_word2index={}
for i, j in enumerate(number_char_vocab):
    number_char_word2index[j] = i
pairs, generate_nums, copy_nums = transfer_num(train_data)
temp_pairs = []
i=0
for p in pairs:
    #if i%10==0:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

for i in [100,200,300,400,500]:
    p=pairs[i]
    print("******")
    print(" ".join(p[0]))
    print(p[1])
    print(p[2])
    print(p[3])

pairs_from_test, _, _ = transfer_num(valid_data)
temp_pairs = []
i=0
for p in pairs_from_test:
    #if i%10==0:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_from_test = temp_pairs

best_acc_fold=[]

input_lang,output_lang,train_pairs,test_pairs,matrix=prepare_APE_data(pairs, pairs_from_test,
    5, generate_nums,copy_nums, tree=True)

pairs_from_valid, _, _ = transfer_num(test_data)
temp_pairs = []
i=0
for p in pairs_from_valid:
    #if i%10==0:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_from_valid = temp_pairs

valid_pairs=prepare_valid_data(input_lang,output_lang,pairs_from_valid, tree=True)


if dataset=="APE":
    print("##############################")
    print("input_lang words"+str(input_lang.n_words))
    print("output_lang words"+str(output_lang.n_words))
    print("generate nums:")
    print(generate_nums)
    print("copy number max nums"+str(copy_nums))
    print("dataset_size:")
    print(len(pairs))
    print(len(pairs_from_test))
    print(len(pairs_from_valid))
    print("dataset_after indexed size:")
    print(len(train_pairs))
    print(len(test_pairs))
    print(len(valid_pairs))

    UNK= output_lang.word2index["UNK"]
    temp_pairs = []
    i=0
    for p in train_pairs:
        if UNK not in p[2]:
            temp_pairs.append(p)
        else:
            i+=1
            if i<5:
                print( " ".join(indexes_to_sentence(input_lang,p[0])))
                print( " ".join(indexes_to_sentence(output_lang,p[2])))
    train_pairs=temp_pairs
    temp_pairs = []
    for p in test_pairs:
        if UNK not in p[2]:
            temp_pairs.append(p)
    test_pairs=temp_pairs
    temp_pairs = []
    for p in valid_pairs:
        if UNK not in p[2]:
            temp_pairs.append(p)
    valid_pairs=temp_pairs

    print("##############################")
    print("dataset_after erase UNK data:")
    print(len(train_pairs))
    print(len(test_pairs))
    print(len(valid_pairs))

for idx_ in [100,200,300,400,500]:
    print("******")
    print( " ".join(indexes_to_sentence(input_lang,train_pairs[idx_][0])))
    print( " ".join(indexes_to_sentence(output_lang,train_pairs[idx_][2])))
    
# Initialize models,here op_nums [PAD, +,- ,*,^,/]
encoder = EncoderAPESeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers,pretrain_emb=matrix)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
pivot=Pivot(hidden_size=hidden_size )
SenLevel=Sentence_level_semantics(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
pivot_optimizer = torch.optim.Adam(pivot.parameters(), lr=learning_rate, weight_decay=weight_decay)
SenLevel_optimizer = torch.optim.Adam(SenLevel.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
predict_scheduler = lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
pivot_scheduler = lr_scheduler.StepLR(pivot_optimizer, step_size=20, gamma=0.5)
SenLevel_scheduler = lr_scheduler.StepLR(SenLevel_optimizer, step_size=20, gamma=0.5)

fold=0
start_epoch=0
last_acc=0.0
best_acc_fold=[[0,0,2316]]


if os.path.exists("models/epoch_num"+str(fold)):
    file_epoch_out=open( "models/epoch_num"+str(fold)).readlines()
    word=file_epoch_out[0]
    start_epoch=int(word.strip())+1
    last_acc_line=file_epoch_out[1]
    last_acc=float(last_acc_line.strip())
    print("start_from_epoch:"+str(start_epoch))
    print("last model acc:"+str(last_acc_line))

    encoder.load_state_dict(torch.load("models/encoder"+str(fold)))
    predict.load_state_dict(torch.load("models/predict"+str(fold)))
    generate.load_state_dict(torch.load("models/generate"+str(fold)))
    merge.load_state_dict(torch.load("models/merge"+str(fold)))
    pivot.load_state_dict(torch.load("models/pivot"+str(fold)))
    SenLevel.load_state_dict(torch.load("models/SenLevel"+str(fold)))
# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    pivot.cuda()
    SenLevel.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, 
    generate_nums,encoder, predict, generate, merge,pivot,SenLevel, encoder_optimizer, predict_optimizer, generate_optimizer,
    merge_optimizer,pivot_optimizer,SenLevel_optimizer, output_lang, num_pos,nums,english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    #nums b*[each_prob_nums]
    nums_index=[]#B*each_prob_nums*num_len
    for b in range(batch_size):
        nums_problem_index=[]
        #print(nums[b])
        for num_word in nums[b]:
            num_word_index=[]
            for num_char in num_word:
                num_word_index.append(number_char_word2index[num_char])
            nums_problem_index.append(num_word_index)
        nums_index.append(nums_problem_index)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    pivot.train()
    SenLevel.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    pivot_optimizer.zero_grad()
    SenLevel_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output,gate_num,num_indicator,just_problem_out,type_num,num_pair_score = encoder(input_var, input_length,nums_index,num_pos)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs,indices,masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    dec_states=[]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask,indices,masked_index)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        curr_nodes=torch.cat((current_embeddings.squeeze(1), current_context.squeeze(1)), 1)#B*1*N
        dec_states.append(curr_nodes)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i- num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()

    dec_states=torch.stack(dec_states,dim=1)

    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    
    num_pos_mask=[]
    for b in range(batch_size):
        num_pos_mask_temp=[0 for _ in range(max_len)]
        num_pos_mask.append(num_pos_mask_temp)
    cate_pos_mask=[]
    for b in range(batch_size):
        cate_pos_mask_temp=[0 for _ in range(max_len)]
        cate_pos_mask.append(cate_pos_mask_temp)

    N1_num_start=num_start+len(generate_nums)
    for b in range(batch_size):
        for i in range(len(target_batch[b])):
            if target_batch[b][i]>=N1_num_start:
                Ni_used=target_batch[b][i]-N1_num_start
                if Ni_used<len(num_pos[b]):
                    num_pos_mask[b][num_pos[b][Ni_used]]=1
                    curr_type=devide_numtype(nums[b][Ni_used])
                    cate_pos_mask[b][num_pos[b][Ni_used]]=curr_type
    

    num_pos_mask=torch.FloatTensor(num_pos_mask).cuda()#B*S
    gate_num_loss = nn.BCELoss()(gate_num, num_pos_mask)


    cate_pos_mask=torch.LongTensor(cate_pos_mask).cuda().reshape(batch_size*max_len)#B*S
    type_num_re=type_num.reshape(batch_size*max_len,5)
    type_num_loss = nn.CrossEntropyLoss()(type_num_re, cate_pos_mask)#B*S*5

    num_compare_mask=[]
    num_compare_score=[]
    max_numpair_len=0.0
    for b in range(batch_size):
        single_num_mask=[]#B*num*num
        #single_num_score=[]
        for i in range(len(num_pos[b])):
            for j in range(len(num_pos[b])):
                pos_a=num_pos[b][i]
                pos_b=num_pos[b][j]
                #num_compare_score.append(num_self_score[b,pos_a,pos_b,:].tolist())#B*S*S*3
                try:
                    nums_a=float(nums[b][i])
                    nums_b=float(nums[b][j])
                    if nums_a>=nums_b:
                        num_compare_mask.append(1)
                        num_compare_score.append(num_pair_score[b,pos_b]-num_pair_score[b,pos_a])
                        max_numpair_len+=1.0
                    else:
                        num_compare_mask.append(-1)
                        num_compare_score.append(num_pair_score[b,pos_a]-num_pair_score[b,pos_b])
                        max_numpair_len+=1.0

                except:
                    single_num_mask=[]
        #num_compare_mask.append(single_num_mask)
        #num_compare_score.append(single_num_score)#B*num*num
    
    #num_compare_mask=torch.FloatTensor(num_compare_mask).cuda()#B*num*num
    num_compare_score=torch.FloatTensor(num_compare_score).cuda()##B*num*num
    #num_compare_score=num_compare_score*num_compare_mask
    zeros = torch.zeros_like(num_compare_score)
    pair_compare_score=torch.where(num_compare_score>0,num_compare_score,zeros)
    pair_compare_sum=torch.sum(pair_compare_score)
    compare_num_loss = torch.div(pair_compare_sum,float(max_numpair_len))#B*S*5
    

    #match_labels = Variable(torch.LongTensor(range(batch_size))).cuda()#B [0,1,2,...,B]
    #attn_score,attn_score1=SenLevel(just_problem_out,current_context)
    #loss0 = nn.CrossEntropyLoss()(attn_score, match_labels)
    #loss1 = nn.CrossEntropyLoss()(attn_score1, match_labels)
    #sent_loss=float(loss0+loss1)/2.0

    #binary_num_pivot=pivot(encoder_outputs,dec_states)
    #binary_num_pivot_loss = nn.BCELoss()(binary_num_pivot, num_indicator)
    loss_total=loss+gate_num_loss+type_num_loss+compare_num_loss
    loss_total.backward()

    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    pivot_optimizer.step()
    SenLevel_optimizer.step()
    del input_var, seq_mask,padding_hidden,num_mask,all_node_outputs,target
    torch.cuda.empty_cache()
    return loss_total.item()  # , loss_0.item(), loss_1.item()

def devide_numtype(num_str):
    #integer, decimal, fraction, percentage 
    num_type=1
    if "%" in num_str:
        num_type=4
    elif "/" in num_str:
        num_type=3
    elif "." in num_str:
        num_type=2
    else:
        num_type=1
    return num_type

def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge,pivot, output_lang,
    num_pos,nums,beam_size=3, english=False, max_length=Max_Expression_len):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    pivot.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1
    #nums b*[each_prob_nums]
    nums_index=[]#B*each_prob_nums*num_len
    for b in range(batch_size):
        nums_problem_index=[]
        for num_word in nums:
            num_word_index=[]
            for num_char in num_word:
                num_word_index.append(number_char_word2index[num_char])
            nums_problem_index.append(num_word_index)
        nums_index.append(nums_problem_index)


    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output,gate_num,num_indicator,just_problem_out,type_num,num_pair_score = encoder(input_var, [input_length],nums_index,[num_pos])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs,indices,masked_index = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask,indices,masked_index)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    del input_var, seq_mask,padding_hidden,num_mask
    torch.cuda.empty_cache()
    return beams[0].out
for epoch in range(start_epoch,n_epochs):
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    pivot_scheduler.step()
    SenLevel_scheduler.step()
    loss_total = 0
    PAD_token=input_lang.word2index["PAD"]
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size,PAD_token)
    print("fold:", fold + 1)
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in range(start_epoch,len(input_lengths)):
        #print(idx)
        loss = train_tree(input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,pivot,SenLevel,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,pivot_optimizer,SenLevel_optimizer, output_lang, num_pos_batches[idx],nums_batches[idx])
        loss_total += loss

    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if epoch % 10 == 0 or epoch > n_epochs - 5:
        out_filename="output/test_result"+str(fold)
        out_filename1="output/test_wrong"+str(fold)
        file_out=open(out_filename,"w")
        file_wrong=open(out_filename1,"w") 
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                     merge,pivot, output_lang, test_batch[5],test_batch[4],beam_size=beam_size)
            val_ac, equ_ac, test_list, tar_list = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if test_list is not None and tar_list is not None:
                file_out.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
            if val_ac:
                value_ac += 1
            else:
                if test_list is not None and tar_list is not None:
                    file_wrong.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        file_out.close()
        file_wrong.close()
        curr_acc=float(value_ac)/eval_total
        if curr_acc>=last_acc:
            torch.save(encoder.state_dict(), "models/encoder"+str(fold))
            torch.save(predict.state_dict(), "models/predict"+str(fold))
            torch.save(generate.state_dict(), "models/generate"+str(fold))
            torch.save(merge.state_dict(), "models/merge"+str(fold))
            torch.save(pivot.state_dict(), "models/pivot"+str(fold))
            torch.save(SenLevel.state_dict(), "models/SenLevel"+str(fold))
            file_epoch_out=open( "models/epoch_num"+str(fold),"w")
            file_epoch_out.write(str(epoch)+"\n")
            file_epoch_out.write(str(curr_acc))
            last_acc=curr_acc
            file_epoch_out.close()
            best_acc_fold[fold][0]=equation_ac
            best_acc_fold[fold][1]=value_ac
            best_acc_fold[fold][2]=eval_total
        if epoch == n_epochs - 1:
            a, b, c = 0, 0, 0
            for bl in range(len(best_acc_fold)):
                print(round(best_acc_fold[bl][0] / float(best_acc_fold[bl][2]),4), round(best_acc_fold[bl][1] / float(best_acc_fold[bl][2]),4))
                a += best_acc_fold[bl][0]
                b += best_acc_fold[bl][1]
                c += best_acc_fold[bl][2]
            print(round(a / float(c),4),round( b / float(c),4))




file_out=open("output/valid_result"+str(fold),"w")
file_wrong=open("output/valid_wrong"+str(fold),"w") 
value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()
for test_batch in valid_pairs:
    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                             merge,pivot, output_lang, test_batch[5] ,test_batch[4],  beam_size=beam_size)
    val_ac, equ_ac, test_list, tar_list = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if test_list is not None and tar_list is not None:
        file_out.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
    if val_ac:
        value_ac += 1
    else:
        if test_list is not None and tar_list is not None:
            file_wrong.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
    if equ_ac:
        equation_ac += 1
    eval_total += 1
print(equation_ac, value_ac, eval_total)
print("final_valid_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("testing time", time_since(time.time() - start))
print("------------------------------------------------------")
file_out.close()
file_wrong.close()

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print("Best Exp:"+str(round(a / float(c),4)))
print("Best Ans:"+str(round( b / float(c),4)))

print("Valid Exp:"+str(round(float(equation_ac) / eval_total,4)))
print("Valid Ans:"+str(round( float(value_ac) / eval_total,4)))
