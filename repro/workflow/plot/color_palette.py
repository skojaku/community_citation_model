import seaborn as sns


def get_palette(data_type):
    if data_type == "aps":
        markercolor = sns.color_palette("pastel", desat=1).as_hex()[0]
        linecolor = sns.color_palette("dark").as_hex()[0]
    elif data_type == "uspto":
        markercolor = sns.color_palette("pastel", desat=1).as_hex()[2]
        linecolor = sns.color_palette("dark").as_hex()[2]
    else:
        markercolor = sns.color_palette("pastel", desat=1).as_hex()[1]
        linecolor = sns.color_palette("dark").as_hex()[1]
    return markercolor, linecolor
