import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {}

model = load_model('model.hdf5')

model.make_predict_function()


def predict_label(img_path):
	img = image.load_img(img_path, target_size=(128, 128))
	img = image.img_to_array(img)/255.0
	img = img.reshape(1, 128, 128, 3)
	p = model.predict(img)
	class_x = np.argmax(p, axis=1)
	maxs = max(p[0])
	if p[0].argmax() == 0:
		predict = "Voiture"
		accuracy = maxs
	elif p[0].argmax() == 1:
		predict = "Chat"
		accuracy = maxs
	else:
		predict = "Chien"
		accuracy = maxs
	print(p)
	return predict, accuracy
	# print(p[0])
	# print(p[0].argmax())
	# print(maxs)
	# print(round(maxs))


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/images/" + img.filename
		img.save(img_path)

		prediction = predict_label(img_path)

	return render_template("index.html", prediction=prediction, img_path=img_path)


if __name__ == '__main__':
	app.run(debug=True)
