def li_match(store_li, brand_li):
    """
    store_li:店铺名称列表
    brand_li:品牌列表
    return：店铺为key， 品牌为value的字典   例：{"立白官方旗舰店"：立白，“舒莱旗舰店”： 舒莱}  如果莫小仙旗舰店没有对应商品，则不存在莫小仙旗舰店这个key
    """
    dic = {}
    for each in brand_li: #遍历每一个品牌
        for every in store_li: # 遍历每一个店铺
            if each in every: # 如果品牌是店铺的子字符串，则添加进字典
                dic[every] = each
    return dic


def get_value(store_li, dic):
    for each in store_li:
        if each not in dic.keys():
            continue
        else:
            value = dic[each]