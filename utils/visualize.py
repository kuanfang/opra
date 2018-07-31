import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64


def plot_annotation(image, points, heatmap, alpha=0.7, fig=None):
    """Visualize the heatmap on the target image.
    """
    if fig is None:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

    # Plot the image.
    image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB)
    ax1.imshow(image)

    # Plot the points.
    ax1.scatter(points[:, 0], points[:, 1], c='r')
    ax1.set_title('points annotation\nn_points: {}'.format(points.shape[0]))

    # Plot the heatmap.
    ax2.imshow(heatmap, cmap='gray', interpolation='nearest')
    ax2.set_title('heatmap\nnp.sum(heatmap): {}'.format(np.sum(heatmap)))

    # Plot the overlay of heatmap on the target image.
    processed_heatmap = heatmap * 255 / np.max(heatmap)
    processed_heatmap = np.tile(processed_heatmap[:, :, np.newaxis], (1, 1, 3))
    processed_heatmap = processed_heatmap.astype('uint8')
    assert processed_heatmap.shape == image.shape
    overlay = cv2.addWeighted(processed_heatmap, alpha, image, 1-alpha, 0)
    ax3.imshow(overlay, interpolation='nearest')
    ax3.set_title('heatmap overlay\nalpha: {}'.format(alpha))


def plot_video(video_path):
    """Visualize the video clip in the Ipython notebook.
    """
    from IPython.display import HTML

    video = io.open(video_path, 'rb').read()
    encoded = base64.b64encode(video)
    video_tag = '''<video width="400" height="222" controls="controls">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                   </video>'''.format(encoded.decode('ascii')) 

    return HTML(data=video_tag)
