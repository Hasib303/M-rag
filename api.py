from fastapi import FastAPI
from app.routes.all_routes import router

app = FastAPI(title="Answer Generator")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Answer Generator API!"}

# Include the single router with all endpoints
app.include_router(router, prefix="/api", tags=["All Services"])

# Run the API
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8000,
        reload=True
    )