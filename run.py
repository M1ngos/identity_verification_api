import uvicorn

if __name__ == "__main__":
    # Run the server
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)