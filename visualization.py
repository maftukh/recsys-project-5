from matplotlib import pyplot as plt

# Visualize metric values for both cases (base/dynamic)
def plot_metric(metric_name,
                stop_times,
                basic_value_history,
                dynamic_value_history,
                n_top=20):
  
    plt.figure(figsize=(12, 6))

    plt.plot(stop_times, basic_value_history, label='basic', marker='o')
    plt.plot(stop_times, dynamic_value_history, label='dynamic', marker='o')

    plt.title(metric_name + '@{}'.format(n_top), fontsize=14)
    plt.legend(fontsize=12)

    plt.xticks(stop_times[::2])
    plt.xlabel('timestamp', fontsize=12)

    plt.grid()
    plt.show()
