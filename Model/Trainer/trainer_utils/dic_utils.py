def update_dic(loss_dic: dict, dic_total: dict) -> None:
    for key in loss_dic.keys():
        if key not in dic_total:
            dic_total[key] = []
        try:
            dic_total[key] += [loss_dic[key].item()]
        except AttributeError:
            dic_total[key] += [loss_dic[key]]
    return dic_total
