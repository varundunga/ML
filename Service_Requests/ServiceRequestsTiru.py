import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
dataset=pd.read_csv('311_Service_Requests_from_2010_to_Present.csv')
data=dataset[['Complaint Type','City']]
#data=data.iloc[0:100,:]

data["counts"] = 1
grp = data.groupby(["City","Complaint Type"])["counts"].sum()
#print(grp)

ucity=data['City'].unique()
ucomplaintdata=data['Complaint Type'].unique()

#print(len(ucity))
#print(len(ucomplaintdata))
#print(grp.head())

data_dict = {"city":[],"comp":[],"count":[]}
for c,cn in grp.iteritems():
    data_dict["city"].append(c[0])
    data_dict["comp"].append(c[1])
    data_dict["count"].append(cn)

new_df = pd.DataFrame.from_dict(data_dict)
#print(new_df.head)
#
#print(len(ucity))
data_count = []
for i,city in enumerate(ucity):
    temp = []
    for comp in ucomplaintdata:
        df = new_df.loc[(new_df['city']==city) & (new_df['comp']==comp)]
        if len(df):
            temp.append(int(df["count"]))
        else:
            temp.append(0)
    data_count.append(temp)

#print(ucity)
#print(ucomplaintdata)
#print(data_count)



def stacked_bar(d, series_labels, category_labels=None, 
                show_values=False, value_format="{}", y_label=None, 
                grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """
    
    #x=data['Complaint Type'].unique()
    ny = len(d[0])
#    print(d[0])
    ind = list(range(ny))
#    print(ny)
#    print(ind)

    axes = []
    cum_size = np.zeros(ny)

    d = np.array(d)

    if reverse:
        d = np.flip(d, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(d):
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i]))
        cum_size += row_data

    if len(category_labels):
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")


stacked_bar(
    data_count, 
    ucity, 
    category_labels=ucomplaintdata, 
    show_values=True, 
    value_format="{:.1f}",
    y_label="Quantity (units)"
)
plt.legend(ucomplaintdata)

plt.savefig('bar.png')
plt.show()
