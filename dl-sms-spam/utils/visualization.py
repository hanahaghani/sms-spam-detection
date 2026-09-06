#import library
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#neural network architecture
def visualize_new_data(text,model,tfidf_vectorizer,save_path):
    tfidf_vector = tfidf_vectorizer.transform([text])
    tfidf_array = tfidf_vector.toarray()

    x = torch.tensor(
        tfidf_array,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():
        hidden1_linear = model.layer1(x)
        hidden1 = F.relu(hidden1_linear)

        hidden2_linear = model.layer2(hidden1)
        hidden2 = F.relu(hidden2_linear)

        hidden3_linear = model.layer3(hidden2)
        hidden3 = F.relu(hidden3_linear)

        output = model.output(hidden3)

        probability = torch.sigmoid(output)

    hidden1_values = hidden1.squeeze(0).cpu().numpy()
    hidden2_values = hidden2.squeeze(0).cpu().numpy()
    hidden3_values = hidden3.squeeze(0).cpu().numpy()

    spam_probability = probability.item()

    feature_names = np.array(
        tfidf_vectorizer.get_feature_names_out()
    )

    tfidf_values = tfidf_array[0]

    top_indices = np.argsort(tfidf_values)[-8:][::-1]

    top_features = feature_names[top_indices]
    top_values = tfidf_values[top_indices]

    top_hidden1_indices = np.argsort(hidden1_values)[-12:][::-1]
    top_hidden2_indices = np.argsort(hidden2_values)[-12:][::-1]
    top_hidden3_indices = np.argsort(hidden3_values)[-12:][::-1]

    hidden1_selected = hidden1_values[top_hidden1_indices]
    hidden2_selected = hidden2_values[top_hidden2_indices]
    hidden3_selected = hidden3_values[top_hidden3_indices]

    fig, ax = plt.subplots(figsize=(18, 9))

    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    x_input = 2
    x_hidden1 = 6
    x_hidden2 = 10
    x_hidden3 = 14
    x_output = 17

    y_positions = np.linspace(2, 8, 12)

    for i, feature_value in enumerate(top_values):
        for j, activation in enumerate(hidden1_selected):
            alpha = min(0.15 + float(feature_value) * float(activation) * 0.2,0.8)

            ax.plot([x_input, x_hidden1],[7.5 - i * 0.7, y_positions[j]],alpha=alpha,linewidth=0.8)

    for i, activation1 in enumerate(hidden1_selected):
        for j, activation2 in enumerate(hidden2_selected):
            alpha = min(0.1 + float(activation1) * float(activation2) * 0.1,0.7)

            ax.plot([x_hidden1, x_hidden2],[y_positions[i], y_positions[j]],alpha=alpha,linewidth=0.7)

    for i, activation2 in enumerate(hidden2_selected):
        for j, activation3 in enumerate(hidden3_selected):
            alpha = min(0.1 + float(activation2) * float(activation3) * 0.1,0.7)

            ax.plot([x_hidden2, x_hidden3],[y_positions[i], y_positions[j]],alpha=alpha,linewidth=0.7)

    for i, activation in enumerate(hidden3_selected):
        alpha = min(0.15 + float(activation) * 0.15,0.9)

        ax.plot([x_hidden3, x_output],[y_positions[i], 5],alpha=alpha,linewidth=1.2)

    for i, (feature, value) in enumerate(zip(top_features, top_values)):
        y = 7.5 - i * 0.7

        ax.scatter(x_input,y,s=250,alpha=min(0.3 + value, 1.0))

        ax.text(x_input - 0.3,y,f"{feature}\n{value:.2f}",ha="right",va="center",fontsize=9)

    for i, activation in enumerate(hidden1_selected):
        ax.scatter(x_hidden1,y_positions[i],s=300,
            alpha=min(0.25 + float(activation) * 0.15,1.0)
        )

        ax.text(x_hidden1,y_positions[i],f"{activation:.2f}",ha="center",va="center",fontsize=8)

    for i, activation in enumerate(hidden2_selected):
        ax.scatter(x_hidden2,y_positions[i],s=300,
            alpha=min(0.25 + float(activation) * 0.15,1.0)
        )

        ax.text(x_hidden2,y_positions[i],f"{activation:.2f}",ha="center",va="center",fontsize=8)

    for i, activation in enumerate(hidden3_selected):
        ax.scatter(x_hidden3,y_positions[i],s=300,
            alpha=min(0.25 + float(activation) * 0.15,1.0)
        )

        ax.text(x_hidden3,y_positions[i],f"{activation:.2f}",ha="center",va="center",fontsize=8)

    ax.scatter(x_output,5,s=500)

    ax.text(x_output,5,f"{spam_probability:.2f}",ha="center",va="center",fontsize=10)

    ax.text(x_input,9.2,"TF-IDF\nTop Features",ha="center",fontsize=12,fontweight="bold")

    ax.text(x_hidden1,9.2,"Hidden Layer 1\n256 neurons",ha="center",fontsize=12,fontweight="bold")

    ax.text(x_hidden2,9.2,"Hidden Layer 2\n128 neurons",ha="center",fontsize=12,fontweight="bold")

    ax.text(x_hidden3,9.2,"Hidden Layer 3\n64 neurons",ha="center",fontsize=12,fontweight="bold")

    ax.text(x_output,9.2,"Output",ha="center",fontsize=12,fontweight="bold")

    ax.text(x_hidden1,0.8,"ReLU + Dropout",ha="center",fontsize=10)

    ax.text(x_hidden2,0.8,"ReLU + Dropout",ha="center",fontsize=10)

    ax.text(x_hidden3,0.8,"ReLU + Dropout",ha="center",fontsize=10)

    ax.text(9,0.2,f'New SMS: "{text}"',ha="center",fontsize=11)

    prediction = "SPAM" if spam_probability >= 0.5 else "HAM"

    ax.text(
        x_output,3.5,f"{prediction}\nP(Spam) = {spam_probability:.2%}",
        ha="center",va="center",fontsize=15,fontweight="bold"
    )

    plt.savefig(save_path,dpi=300,bbox_inches="tight")

    plt.close()

#training loss vs validaition loss
def plot_training_history(training_losses,valid_losses,save_path):

    plt.figure(figsize=(10,8))
    epochs=range(1,len(training_losses)+1)
    plt.plot(epochs,training_losses,label="training loss")
    plt.plot(epochs,valid_losses,label='validation loss')
    plt.xlabel=('Epoch')
    plt.ylabel=('loss')
    plt.title("training and validation loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

