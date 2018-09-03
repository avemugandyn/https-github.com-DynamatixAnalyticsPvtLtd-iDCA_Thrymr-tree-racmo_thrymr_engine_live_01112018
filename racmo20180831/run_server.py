from web.app import create_app

app,model = create_app()
if __name__ == '__main__':
    app.run(port=5050, debug=False, host="188.165.246.32")
