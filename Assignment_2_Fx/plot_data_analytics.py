import matplotlib.pyplot as plt
import pandas as pd


def plot_data_analytics():

    path = 'Assignment_2_Fx/Assignment_2_Data_sets/stats/k_means_simple_results.csv'
    k_simple = pd.read_csv(path)

    path = 'Assignment_2_Fx/Assignment_2_Data_sets/stats/k_means_k_anonymity_results.csv'
    k_anonymity = pd.read_csv(path)

    path = 'Assignment_2_Fx/Assignment_2_Data_sets/stats/k_means_l_diversity_results.csv'
    l_diversity = pd.read_csv(path)

    l_diversity = l_diversity.sort_values(by='Size').reset_index(drop=True)

    l_diversity['Time'] += k_simple['Time']

    path = 'Assignment_2_Fx/Assignment_2_Data_sets/stats/differential_privacy_results.csv'
    dp_data = pd.read_csv(path)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(dp_data['Size'], dp_data['Time'], label='Time', marker='o', color='tab:blue')
    ax1.set_xlabel('Size')
    ax1.set_ylabel('Time', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(dp_data['Size'], dp_data['Info_Loss'], label='Info Loss', marker='o', color='tab:red')
    ax2.plot(dp_data['Size'], dp_data['Q_Accuracy'], label='Q Accuracy', marker='o', color='tab:green')
    ax2.set_ylabel('Info Loss & Q Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Dataset Results with Separate Scales: Time, Info Loss, and Q Accuracy vs Size')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/stats/diff_privacy_results.png")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(k_simple['Size'], k_simple['Time'], label='Time', marker='o', color='tab:blue')
    ax1.set_xlabel('Size')
    ax1.set_ylabel('Time', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(k_simple['Size'], k_simple['Info_Loss'], label='Info Loss', marker='o', color='tab:red')
    ax2.plot(k_simple['Size'], k_simple['Q_Accuracy'], label='Q Accuracy', marker='o', color='tab:green')
    ax2.set_ylabel('Info Loss & Q Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.title('Dataset Results with Separate Scales: Time, Info Loss, and Q Accuracy vs Size')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/stats/k_means_simple_results.png")


    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(k_anonymity['Size'], k_anonymity['Time'], label='Time', marker='o', color='tab:blue')
    ax1.set_xlabel('Size')
    ax1.set_ylabel('Time', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(k_anonymity['Size'], k_anonymity['Info_Loss'], label='Info Loss', marker='o', color='tab:red')
    ax2.plot(k_anonymity['Size'], k_anonymity['Q_Accuracy'], label='Q Accuracy', marker='o', color='tab:green')
    ax2.set_ylabel('Info Loss & Q Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.title('Dataset Results with Separate Scales: Time, Info Loss, and Q Accuracy vs Size')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/stats/k_means_k_anonymity_results.png")


    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(l_diversity['Size'], l_diversity['Time'], label='Time', marker='o', color='tab:blue')
    ax1.set_xlabel('Size')
    ax1.set_ylabel('Time', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(l_diversity['Size'], l_diversity['Info_Loss'], label='Info Loss', marker='o', color='tab:red')
    ax2.plot(l_diversity['Size'], l_diversity['Q_Accuracy'], label='Q Accuracy', marker='o', color='tab:green')
    ax2.set_ylabel('Info Loss & Q Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.title('Dataset Results with Separate Scales: Time, Info Loss, and Q Accuracy vs Size')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/stats/k_means_l_diversity_results.png")


    path = 'Assignment_2_Fx/Assignment_2_Data_sets/stats/ucc_handler_results.csv'
    ucc_handler = pd.read_csv(path)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(ucc_handler['Size'], ucc_handler['Time'], label='Time', marker='o', color='tab:blue')
    ax1.set_xlabel('Size')
    ax1.set_ylabel('Time', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(ucc_handler['Size'], ucc_handler['Info_Loss'], label='Info Loss', marker='o', color='tab:red')
    ax2.plot(ucc_handler['Size'], ucc_handler['Q_Accuracy'], label='Q Accuracy', marker='o', color='tab:green')
    ax2.set_ylabel('Info Loss & Q Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.title('Dataset Results with Separate Scales: Time, Info Loss, and Q Accuracy vs Size')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/stats/ucc_handler_results.png")

