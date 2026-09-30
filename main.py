import argparse
from ingest import ingest
from query import query

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='cmd')

p1 = sub.add_parser('ingest')
p1.add_argument('file')
p1.add_argument('--outdir', default='data', help='Output directory for chunks/embeddings')
p1.add_argument('--model', default=None, help='Embedding model name (overrides default)')
p1.add_argument('--chunk-size', type=int, default=None, help='Chunk size (words)')
p1.add_argument('--overlap', type=int, default=None, help='Chunk overlap (words)')

p2 = sub.add_parser('query')
p2.add_argument('q', help='Question to ask')
p2.add_argument('--k', type=int, default=5, help='Number of context chunks to retrieve')
p2.add_argument('--embed-model', default=None, help='Embedding model to use')
p2.add_argument('--rag-model', default=None, help='LLM model for synthesis')
p2.add_argument('--data-dir', default='data', help='Directory where embeddings/chunks live')

args = parser.parse_args()
if args.cmd == 'ingest':
    ingest(args.file, out_dir=args.outdir, model_name=(args.model if args.model else None) or None,
           chunk_size=(args.chunk_size if args.chunk_size is not None else None) or None,
           overlap=(args.overlap if args.overlap is not None else None) or None)
elif args.cmd == 'query':
    embed_model = args.embed_model if args.embed_model else None
    rag_model = args.rag_model if args.rag_model else None
    print(query(args.q, k=args.k, embed_model=(embed_model or None), rag_model=(rag_model or None), data_dir=args.data_dir))
else:
    parser.print_help()
