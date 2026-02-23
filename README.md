# Lotomania Previsões Inteligentes

Este projeto utiliza inteligência artificial para gerar previsões de 50 números para a Lotomania, com interface web responsiva e possibilidade de ajuste de hiperparâmetros.

## Principais arquivos
- `app_flask_lotomania.py`: Backend Flask do aplicativo web
- `templates/index.html`: Interface web responsiva
- `predictor_tiinew05.py`: Lógica de geração de previsões
- `modelo_tii_superorganismo.pkl`: Modelo treinado (opcional, para previsões rápidas)
- `loto_mania_asloterias_ate_concurso_2835_sorteio.xlsx`: Dados históricos
- `requirements.txt`: Dependências do projeto
- `Procfile`: Comando de inicialização para deploy na Render

## Como rodar localmente
1. Instale as dependências:
	```
	pip install -r requirements.txt
	```
2. Execute o app Flask:
	```
	python app_flask_lotomania.py
	```
3. Acesse `http://localhost:5000` no navegador.

## Como publicar na nuvem (Render)
1. Suba o projeto para um repositório no GitHub.
2. Crie uma conta em [Render](https://render.com).
3. Crie um novo Web Service, conecte ao repositório e configure:
	- Build Command: `pip install -r requirements.txt`
	- Start Command: `gunicorn app_flask_lotomania:app`
4. O app estará disponível em um link público.

## Observações
- A pasta `__pycache__` não é necessária para o deploy.
- A pasta `snapshots` só é necessária se algum script depender dela.
- O arquivo `modelo_tii_superorganismo.pkl` acelera as previsões, mas será criado automaticamente se não existir.

## Licença
Projeto para fins educacionais e experimentais.
