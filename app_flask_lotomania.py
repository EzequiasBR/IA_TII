from flask import Flask, render_template, request
import subprocess
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    if request.method == 'POST':
        robos = request.form.get('robos', 150)
        geracoes = request.form.get('geracoes', 10)
        mutacao = request.form.get('mutacao', 0.03)
        top_percent = request.form.get('top_percent', 0.3)
        previsoes = request.form.get('previsoes', 5)
        cmd = f"python predictor_tiinew05.py --robos {robos} --geracoes {geracoes} --mutacao {mutacao} --top_percent {top_percent} --previsoes {previsoes}"
        resultado = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
