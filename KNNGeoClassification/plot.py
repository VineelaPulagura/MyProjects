# Created By    : Tolga
# Created on    : 16-07-2021
# Description   : To plot the dataset to understand proximity of the street sites with GPS coordinates

import matplotlib.pyplot as plt
import numpy as np
import re


class Plot:
    def plot_latlon_street(self):
        path = "Logs\\KNN_plots"
        for StreetName in ["WideStreet", "MiddleWideStreet", "NarrowStreet"]:
            fig = plt.figure(figsize=(15, 8))
            plt.suptitle('Latitude and Longitude data of the ' + StreetName, fontsize=18)
            plt.xlabel('Latitude', fontsize=18)
            plt.ylabel('Longitude', fontsize=18)
            for x in ["LeftSide", "RightSide"]:
                InputFile = 'Dataset\\StreetData\\' + StreetName + '\\' + x + '\\' + x + '.gpx'
                data = open(InputFile).read()
                lat = np.array(re.findall(r'lat="([^"]+)', data), dtype=float)
                lon = np.array(re.findall(r'lon="([^"]+)', data), dtype=float)
                plt.ticklabel_format(useOffset=False)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.scatter(lat, lon, label=x)
                if StreetName == "WideStreet":
                    plt.xlim(51.308, 51.310)
                if StreetName == "MiddleWideStreet":
                    plt.xlim(51.305, 51.307)
                if StreetName == "NarrowStreet":
                    plt.xlim(51.311, 51.313)
                plt.ylim(9.448, 9.470)
            plt.legend()
            plt.show()
            fig.savefig(f"{path}\\{StreetName}.png")
            plt.close()


if __name__ == '__main__':
    plot = Plot()
    plot.plot_latlon_street()
