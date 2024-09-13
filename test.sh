echo "About to perform load..."

curl -X POST http://127.0.0.1:5000/vectorstore \
  -H "Content-Type: application/json" \
  -d '{
    "type": "load",
    "text": "I am thirsty and want a drink of water",
    "collection": "amygdala"
  }'

echo "About to perform search..."

curl -X POST http://127.0.0.1:5000/vectorstore \
  -H "Content-Type: application/json" \
  -d '{
    "type": "search",
    "query": "I want a drink",
    "collection": "amygdala"
  }'
