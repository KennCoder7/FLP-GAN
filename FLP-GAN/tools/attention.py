import torch
import torch.nn as nn


def func_attention(query, context, gamma1, wMasks, imMasks):
    """ Get region context according to word relevance
    Args:
        query: words feature (batch x ndf x queryL)
        context: region feature (batch x ndf x 5)
        gamma1: smooth hyper param
        imMasks: roi mask (batch_size x ih x iw)
        wMasks: stop words (1 x queryL)
    Returns:
        weightedContext: word's region context (batch x ndf x queryL)
        attn:  word-region att map
    """
    batch_size, queryL = query.size(0), query.size(2)
    if len(context.shape) > 3:
        ih, iw = context.size(2), context.size(3)
        sourceL = ih * iw
    else:
        sourceL = 5
        # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention(similarity)
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)

    # # mask useless regions
    # attn.data.masked_fill_(imMasks, -float("inf"))

    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)

    ## fill stopwords with -inf
    if wMasks is not None:
        # --> batch*sourceL x queryL
        wMasks = wMasks.repeat(batch_size * sourceL, 1)
        attn.data.masked_fill_(wMasks.bool(), -float("inf"))

    attn = nn.functional.softmax(attn, dim=1)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9) in AttnGAN paper
    attn = attn * gamma1

    # fill useless regions with -inf
    if imMasks is not None:
        # --> batch x 1 x sourceL
        imMasks = imMasks.view(batch_size, -1, sourceL)
        # --> batch*queryL x sourceL
        imMasks = imMasks.repeat(1, queryL, 1)
        imMasks = imMasks.view(batch_size * queryL, sourceL)
        attn.data.masked_fill_(imMasks, -float("inf"))
    attn = nn.functional.softmax(attn, dim=1)

    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    if sourceL == 5:
        attn = attn.view(batch_size, -1, 5)
    else:
        attn = attn.view(batch_size, -1, ih, iw)
    return weightedContext, attn
