from collections import Counter



def get_bias(S_ba , S_bc, dataloader):

    S_ba = list(S_ba)
    S_bc = list(S_bc)

    bias_list_ba = []  # Initialize an empty list to store bias values
    bias_list_bc = []
    for element in S_ba:
        bias = dataloader.dataset.get_additional_info(idx=element, column_name='sex')
        bias_list_ba.append(bias)

    for element in S_bc:
        bias = dataloader.dataset.get_additional_info(idx=element, column_name='sex')
        bias_list_bc.append(bias)

    # Count occurrences of 'male' and 'female' in bias_list
    bias_counter_ba = Counter(bias_list_ba)
    male_count_ba = bias_counter_ba['male']
    female_count_ba = bias_counter_ba['female']

    bias_counter_bc = Counter(bias_list_bc)
    male_count_bc = bias_counter_bc['male']
    female_count_bc = bias_counter_bc['female']

    print(1)


