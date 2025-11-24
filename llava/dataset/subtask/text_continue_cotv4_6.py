from .pred_textnoc_cotv4_6 import *
cate_dict = {x: i for i, x in enumerate(('title', 'subtitle', 'bodytext', 'calls to action', 'detailed items', 'menu items', 'social media', 'date', 'name', 'website', 'phone number', 'location'))}
def simple_order(pd):
    pd = sorted(pd, key=lambda x:cate_dict.get(x.lower(), 100))
    return pd
def convert(pk, flip):
    ratio = np.random.rand()
    flip = False
    bg = pk[0]
    bgimg = download_image(bg)
    layers, gua = convert_dct_list_text_buttons(pk)
    layers = [x for x in layers if total_valid(x, bgimg.size)]
    uf = CustomUnionFind(hash_function=lambda x: (json.dumps(x), tuple(x['bbox'])), compare_function=lambda x, y: do_overlap(x[1],y[1]))
    uf.initialize(layers)
    grp = uf.groups()
    layers = [give_order_text([json.loads(y[0]) for y in x])[0] for x in grp]
    layers = give_order_text(layers)
    l = []
    for x in layers:
        dct = {}
        for k, v in x.items():
            if k == 'Text':
                dct['text'] = v
            dct[k] = v
        l.append(dct)
    layers = l
    if len(layers) == 0:
        raise Exception('no valid layer')
    for x in layers:
        x['charsize'] = calc_charsize(x, bgimg.size[:2])
    num = len(layers)
    done = min(int(num*ratio), num-1)

    donelayers_ind = np.random.choice(range(num), done, replace=False)
    donelayers = [x for i, x in enumerate(layers) if i in donelayers_ind]
    notdonelayers = [x for i, x in enumerate(layers) if i not in donelayers_ind]
    inpimg = draw_all_button(bgimg, donelayers, flip)
    _ = draw_all_button(bgimg, notdonelayers, flip)


    first = layers[0]["text"]
    but = layers[0]['buttons']
    dc = []
    used = []
    if len(donelayers) == 0:
        ppp = 'Nothing has been drawn.\n'
    else:
        for x in donelayers:
            but = json.dumps(x['buttons'])
            for y in x['buttons']:
                used.append(json.dumps(y))
            dc.append(json.dumps(x['bbox'])+ f"(underlay: {but})")
        ppp = 'Text blocks have been drawn at: ' + ', '.join(dc) + '.\n'
    used = set(used)

    layers = notdonelayers
    npimg = np.array(inpimg)
    width, height = bgimg.size
    bboxes = [ly['Bounding Box'] for ly in layers]
    centers = [((bbox[0]+bbox[2])//2,(bbox[1]+bbox[3])//2) for bbox in bboxes]
    centers = [convert_bucket(x, 30, bgimg.size[:2]) for x in centers]

    ###########
    org_brightness = []
    for x in layers:
        
        x0,y0,x1,y1 = x['Bounding Box']
        org_brightness.append(judge_color(np.array(rgba2rgb(Image.fromarray(npimg[y0:y1, x0:x1]))).reshape(-1, 3).mean(axis=0)))
        x['brightness'] = org_brightness[-1]
    # return layers
    for tid, _ in enumerate(layers):
        x, y = centers[tid]
        vpos = get_position(y, height, 'height')
        hpos = get_position(x, width, 'width')
        abs_phrase = 'at ' + get_composition([_,], True)# get_absolute_phrase(vpos, hpos)
        rel_phrase = ''
        if tid == 0:
            s = f"Text {tid} should be {abs_phrase} of the poster."
        else:
            rel_phrase = get_relative_phrase(centers[tid-1], centers[tid])
            s = f"Text {tid} should be {abs_phrase} of the poster, and {rel_phrase} of Text {tid-1}."
        layers[tid]['vpos'] = vpos
        layers[tid]['hpos'] = hpos
        layers[tid]['rel'] = rel_phrase
        layers[tid]['abs'] = abs_phrase
    # return layers
    layout = get_composition(layers)
    first = layers[0]["text"]
    fbs = layers[0]['charsize']
    but = layers[0]['buttons']

    if len(but) == 0:
        th = 'no underlay is needed, '
    elif len(but) == 1:
        if json.dumps(but[0]) in used:
            th = f"placed on used underlay {but[0]}, "
        else:
            th = f"placed on unused underlay {but[0]}, "
    else:
        ff = ', '.join([json.dumps(x) for x in but])
        fla = False
        for z in but:
            if json.dumps(z) in used:
                fla = True
                break
        if fla:
            th = f"placed among some used underlays {ff}, "
        else:
            th = f"placed among unused underlays {ff}, "
    # cot = f"Firstly, the first {first}'s boxsize should be {fbs}, and be placed " + layers[0]['abs'] + ' globally. The brightness of that background is ' + layers[0]['brightness'] + '. '
    cot = f"The first <{first}> " + th + layers[0]['abs'] + ' globally. The brightness of that background is ' + layers[0]['brightness'] + '.\n'
    for x in layers[1:]:
        if x["text"] == first:
            sw = 'The next'
        else:
            sw = 'The first'
        first = x["text"]
        bs = x['charsize']
        but = x['buttons']
        if len(but) == 0:
            th = 'no underlay is needed, '
        elif len(but) == 1:
            if json.dumps(but[0]) in used:
                th = f"placed on used underlay {but[0]}, "
            else:
                th = f"placed on unused underlay {but[0]}, "
        else:
            ff = ', '.join([json.dumps(x) for x in but])
            fla = False
            for z in but:
                if json.dumps(z) in used:
                    fla = True
                    break
            if fla:
                th = f"placed among some used underlays {ff}, "
            else:
                th = f"placed among unused underlays {ff}, "
        cot += (f"{sw} <{first}> " + th + x['abs'] + ' globally. The brightness of that background is ' + x['brightness'] + '.\n')
        # x['thought'] = th + x['abs'] + ' globally. The brightness of that background is ' + x['brightness'] + '. '
    [[x.pop('vpos'), x.pop('thought'), x.pop('hpos'), x.pop('rel'), x.pop('charsize'), x.pop('abs'), x.pop('img'), x.pop("Bounding Box"), x.pop("brightness"), x.pop("buttons"), x.pop("Text"), x.pop("category")] for x in layers]
    iper = add_newline(json.dumps([{"text": x["text"] , "char_num":x["char_num"]} for x in layers]))
    [x.pop('text') for x in layers]
    gua = [x for x in gua if json.dumps(x) not in used]
    btg = f'Detected {len(gua)} unused underlays' + ' '*(len(gua)>0) + ', '.join([json.dumps(x) for x in gua]) + '.\n'


    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a half-finished poster image and a series of text to be added to the poster subsequently, predict the metadata for each text metadata listed below. Think step by step.''' + \
        f'''\ninput: {iper}'''},
        {
            'from': 'gpt',
            'value': ppp + btg + cot + '\n' +add_newline(json.dumps(layers))
        }
    ]
    return dialog, inpimg