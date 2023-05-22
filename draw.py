import matplotlib.pyplot as plt


def draw(x, y, title):
    size = 14
    plt.plot(x, y)
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.xlabel('Number of variables', fontsize=size)
    plt.ylabel('Time (seconds)', fontsize=size)
    plt.title(title, fontsize=size)
    plt.savefig(f'pic/{title}.svg', format='svg')
    plt.close()


if __name__ == '__main__':
    x = [i for i in range(100, 650, 50)]
    y = [0.06173, 0.10296, 0.1304,	0.15541,	0.1835,	0.22407,
         0.24932,	0.27834,	0.30754,	0.34576,	0.37785]
    draw(x, y, 'MPR')

    x = [i for i in range(100, 400, 50)]
    y = [0.00364,	0.02814,	0.29888,	2.77901,	22.72629,	350.47689]

    draw(x, y, 'Glucose')
