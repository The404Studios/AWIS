using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.IO.Compression;

namespace AWIS.NLP
{
    #region Byte-Pair Encoding Tokenizer

    public class BPETokenizer
    {
        private readonly Dictionary<string, int> vocabulary;
        private readonly Dictionary<int, string> reverseVocabulary;
        private readonly Dictionary<(string, string), int> bpeMerges;
        private readonly int vocabSize;

        public BPETokenizer(int vocabSize = 10000)
        {
            this.vocabSize = vocabSize;
            this.vocabulary = new Dictionary<string, int>();
            this.reverseVocabulary = new Dictionary<int, string>();
            this.bpeMerges = new Dictionary<(string, string), int>();
            InitializeSpecialTokens();
        }

        private void InitializeSpecialTokens()
        {
            var tokens = new[] { "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>" };
            for (int i = 0; i < tokens.Length; i++)
            {
                vocabulary[tokens[i]] = i;
                reverseVocabulary[i] = tokens[i];
            }
        }

        public void Train(List<string> corpus, int minFrequency = 2)
        {
            // Initialize with character-level vocabulary
            var charFreq = new Dictionary<string, int>();

            foreach (var text in corpus)
            {
                foreach (char c in text)
                {
                    string charStr = c.ToString();
                    if (!charFreq.ContainsKey(charStr))
                        charFreq[charStr] = 0;
                    charFreq[charStr]++;
                }
            }

            // Add characters to vocabulary
            int idx = vocabulary.Count;
            foreach (var kvp in charFreq.Where(x => x.Value >= minFrequency).OrderByDescending(x => x.Value))
            {
                if (!vocabulary.ContainsKey(kvp.Key))
                {
                    vocabulary[kvp.Key] = idx;
                    reverseVocabulary[idx] = kvp.Key;
                    idx++;
                }
            }

            // Learn BPE merges
            var wordFreq = new Dictionary<string, int>();
            foreach (var text in corpus)
            {
                var words = text.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                foreach (var word in words)
                {
                    if (!wordFreq.ContainsKey(word))
                        wordFreq[word] = 0;
                    wordFreq[word]++;
                }
            }

            while (vocabulary.Count < vocabSize)
            {
                var pairFreq = GetPairFrequencies(wordFreq);
                if (pairFreq.Count == 0)
                    break;

                var bestPair = pairFreq.OrderByDescending(x => x.Value).First().Key;
                string merged = bestPair.Item1 + bestPair.Item2;

                if (!vocabulary.ContainsKey(merged))
                {
                    vocabulary[merged] = idx;
                    reverseVocabulary[idx] = merged;
                    idx++;
                }

                bpeMerges[bestPair] = vocabulary[merged];
                wordFreq = ApplyMerge(wordFreq, bestPair, merged);
            }
        }

        private Dictionary<(string, string), int> GetPairFrequencies(Dictionary<string, int> wordFreq)
        {
            var pairFreq = new Dictionary<(string, string), int>();

            foreach (var kvp in wordFreq)
            {
                var tokens = SplitWord(kvp.Key);
                for (int i = 0; i < tokens.Count - 1; i++)
                {
                    var pair = (tokens[i], tokens[i + 1]);
                    if (!pairFreq.ContainsKey(pair))
                        pairFreq[pair] = 0;
                    pairFreq[pair] += kvp.Value;
                }
            }

            return pairFreq;
        }

        private List<string> SplitWord(string word)
        {
            var tokens = new List<string>();
            foreach (char c in word)
            {
                tokens.Add(c.ToString());
            }
            return tokens;
        }

        private Dictionary<string, int> ApplyMerge(Dictionary<string, int> wordFreq, (string, string) pair, string merged)
        {
            var newWordFreq = new Dictionary<string, int>();

            foreach (var kvp in wordFreq)
            {
                string newWord = kvp.Key.Replace(pair.Item1 + pair.Item2, merged);
                newWordFreq[newWord] = kvp.Value;
            }

            return newWordFreq;
        }

        public List<int> Encode(string text)
        {
            var tokens = new List<int> { vocabulary["<BOS>"] };
            var words = text.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);

            foreach (var word in words)
            {
                var wordTokens = EncodeWord(word);
                tokens.AddRange(wordTokens);
            }

            tokens.Add(vocabulary["<EOS>"]);
            return tokens;
        }

        private List<int> EncodeWord(string word)
        {
            var tokens = SplitWord(word);
            bool changed = true;

            while (changed)
            {
                changed = false;
                for (int i = 0; i < tokens.Count - 1; i++)
                {
                    var pair = (tokens[i], tokens[i + 1]);
                    if (bpeMerges.ContainsKey(pair))
                    {
                        string merged = tokens[i] + tokens[i + 1];
                        tokens[i] = merged;
                        tokens.RemoveAt(i + 1);
                        changed = true;
                        break;
                    }
                }
            }

            var ids = new List<int>();
            foreach (var token in tokens)
            {
                if (vocabulary.ContainsKey(token))
                {
                    ids.Add(vocabulary[token]);
                }
                else
                {
                    ids.Add(vocabulary["<UNK>"]);
                }
            }

            return ids;
        }

        public string Decode(List<int> tokenIds)
        {
            var tokens = new List<string>();

            foreach (var id in tokenIds)
            {
                if (id == vocabulary["<BOS>"] || id == vocabulary["<EOS>"] || id == vocabulary["<PAD>"])
                    continue;

                if (reverseVocabulary.ContainsKey(id))
                {
                    tokens.Add(reverseVocabulary[id]);
                }
            }

            return string.Join("", tokens);
        }

        public void Save(string path)
        {
            using (var writer = new StreamWriter(path))
            {
                writer.WriteLine("VOCABULARY");
                foreach (var kvp in vocabulary)
                {
                    writer.WriteLine($"{kvp.Key}\t{kvp.Value}");
                }

                writer.WriteLine("MERGES");
                foreach (var kvp in bpeMerges)
                {
                    writer.WriteLine($"{kvp.Key.Item1}\t{kvp.Key.Item2}\t{kvp.Value}");
                }
            }
        }

        public void Load(string path)
        {
            vocabulary.Clear();
            reverseVocabulary.Clear();
            bpeMerges.Clear();

            using (var reader = new StreamReader(path))
            {
                string section = "";
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    if (line == "VOCABULARY" || line == "MERGES")
                    {
                        section = line;
                        continue;
                    }

                    if (section == "VOCABULARY")
                    {
                        var parts = line.Split('\t');
                        if (parts.Length == 2 && int.TryParse(parts[1], out int id))
                        {
                            vocabulary[parts[0]] = id;
                            reverseVocabulary[id] = parts[0];
                        }
                    }
                    else if (section == "MERGES")
                    {
                        var parts = line.Split('\t');
                        if (parts.Length == 3 && int.TryParse(parts[2], out int id))
                        {
                            bpeMerges[(parts[0], parts[1])] = id;
                        }
                    }
                }
            }
        }
    }

    #endregion

    #region WordPiece Tokenizer

    public class WordPieceTokenizer
    {
        private readonly Dictionary<string, int> vocabulary;
        private readonly Dictionary<int, string> reverseVocabulary;
        private readonly int maxInputChars;
        private readonly string unkToken;
        private readonly string continuePrefix;

        public WordPieceTokenizer(int maxInputChars = 200)
        {
            this.maxInputChars = maxInputChars;
            this.vocabulary = new Dictionary<string, int>();
            this.reverseVocabulary = new Dictionary<int, string>();
            this.unkToken = "[UNK]";
            this.continuePrefix = "##";
            InitializeSpecialTokens();
        }

        private void InitializeSpecialTokens()
        {
            var tokens = new[] { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]" };
            for (int i = 0; i < tokens.Length; i++)
            {
                vocabulary[tokens[i]] = i;
                reverseVocabulary[i] = tokens[i];
            }
        }

        public void BuildVocabulary(List<string> corpus, int vocabSize = 30000)
        {
            var wordFreq = new Dictionary<string, int>();

            foreach (var text in corpus)
            {
                var words = text.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                foreach (var word in words)
                {
                    if (!wordFreq.ContainsKey(word))
                        wordFreq[word] = 0;
                    wordFreq[word]++;
                }
            }

            int idx = vocabulary.Count;

            // Add whole words first
            foreach (var kvp in wordFreq.OrderByDescending(x => x.Value).Take(vocabSize / 2))
            {
                if (!vocabulary.ContainsKey(kvp.Key))
                {
                    vocabulary[kvp.Key] = idx;
                    reverseVocabulary[idx] = kvp.Key;
                    idx++;
                }
            }

            // Add subword pieces
            var subwordFreq = new Dictionary<string, int>();
            foreach (var word in wordFreq.Keys)
            {
                for (int len = 2; len <= Math.Min(word.Length, 10); len++)
                {
                    for (int start = 0; start <= word.Length - len; start++)
                    {
                        string subword = word.Substring(start, len);
                        if (start > 0)
                            subword = continuePrefix + subword;

                        if (!subwordFreq.ContainsKey(subword))
                            subwordFreq[subword] = 0;
                        subwordFreq[subword] += wordFreq[word];
                    }
                }
            }

            foreach (var kvp in subwordFreq.OrderByDescending(x => x.Value))
            {
                if (vocabulary.Count >= vocabSize)
                    break;

                if (!vocabulary.ContainsKey(kvp.Key))
                {
                    vocabulary[kvp.Key] = idx;
                    reverseVocabulary[idx] = kvp.Key;
                    idx++;
                }
            }
        }

        public List<int> Tokenize(string text)
        {
            var tokens = new List<int> { vocabulary["[CLS]"] };
            var words = text.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);

            foreach (var word in words)
            {
                if (word.Length > maxInputChars)
                {
                    tokens.Add(vocabulary[unkToken]);
                    continue;
                }

                var subTokens = TokenizeWord(word);
                tokens.AddRange(subTokens);
            }

            tokens.Add(vocabulary["[SEP]"]);
            return tokens;
        }

        private List<int> TokenizeWord(string word)
        {
            var tokens = new List<int>();
            int start = 0;

            while (start < word.Length)
            {
                int end = word.Length;
                int? foundId = null;

                while (start < end)
                {
                    string substr = word.Substring(start, end - start);
                    if (start > 0)
                        substr = continuePrefix + substr;

                    if (vocabulary.ContainsKey(substr))
                    {
                        foundId = vocabulary[substr];
                        break;
                    }

                    end--;
                }

                if (foundId == null)
                {
                    tokens.Add(vocabulary[unkToken]);
                    return tokens;
                }

                tokens.Add(foundId.Value);
                start = end;
            }

            return tokens;
        }

        public string Detokenize(List<int> tokenIds)
        {
            var tokens = new List<string>();

            foreach (var id in tokenIds)
            {
                if (id == vocabulary["[CLS]"] || id == vocabulary["[SEP]"] || id == vocabulary["[PAD]"])
                    continue;

                if (reverseVocabulary.ContainsKey(id))
                {
                    string token = reverseVocabulary[id];
                    tokens.Add(token);
                }
            }

            // Join tokens and remove continue prefixes
            var result = string.Join(" ", tokens);
            result = result.Replace(" " + continuePrefix, "");
            return result;
        }
    }

    #endregion

    #region Compressed Tokenizer with Huffman Coding

    public class HuffmanNode
    {
        public int Frequency { get; set; }
        public int? TokenId { get; set; }
        public HuffmanNode? Left { get; set; }
        public HuffmanNode? Right { get; set; }

        public bool IsLeaf => TokenId.HasValue;
    }

    public class CompressedTokenizer
    {
        private readonly BPETokenizer baseTokenizer;
        private readonly Dictionary<int, string> huffmanCodes;
        private HuffmanNode? huffmanTree;
        private readonly Dictionary<int, int> tokenFrequencies;

        public CompressedTokenizer(int vocabSize = 10000)
        {
            this.baseTokenizer = new BPETokenizer(vocabSize);
            this.huffmanCodes = new Dictionary<int, string>();
            this.tokenFrequencies = new Dictionary<int, int>();
        }

        public void Train(List<string> corpus)
        {
            // Train base tokenizer
            baseTokenizer.Train(corpus);

            // Compute token frequencies
            tokenFrequencies.Clear();
            foreach (var text in corpus)
            {
                var tokens = baseTokenizer.Encode(text);
                foreach (var token in tokens)
                {
                    if (!tokenFrequencies.ContainsKey(token))
                        tokenFrequencies[token] = 0;
                    tokenFrequencies[token]++;
                }
            }

            // Build Huffman tree
            BuildHuffmanTree();
            GenerateHuffmanCodes();
        }

        private void BuildHuffmanTree()
        {
            var nodes = new List<HuffmanNode>();

            foreach (var kvp in tokenFrequencies)
            {
                nodes.Add(new HuffmanNode
                {
                    TokenId = kvp.Key,
                    Frequency = kvp.Value
                });
            }

            while (nodes.Count > 1)
            {
                nodes = nodes.OrderBy(n => n.Frequency).ToList();

                var left = nodes[0];
                var right = nodes[1];

                var parent = new HuffmanNode
                {
                    Frequency = left.Frequency + right.Frequency,
                    Left = left,
                    Right = right
                };

                nodes.RemoveAt(0);
                nodes.RemoveAt(0);
                nodes.Add(parent);
            }

            huffmanTree = nodes.Count > 0 ? nodes[0] : null;
        }

        private void GenerateHuffmanCodes()
        {
            huffmanCodes.Clear();
            if (huffmanTree != null)
            {
                GenerateCodesRecursive(huffmanTree, "");
            }
        }

        private void GenerateCodesRecursive(HuffmanNode node, string code)
        {
            if (node.IsLeaf)
            {
                huffmanCodes[node.TokenId.Value] = code.Length > 0 ? code : "0";
                return;
            }

            if (node.Left != null)
                GenerateCodesRecursive(node.Left, code + "0");

            if (node.Right != null)
                GenerateCodesRecursive(node.Right, code + "1");
        }

        public byte[] EncodeCompressed(string text)
        {
            var tokens = baseTokenizer.Encode(text);
            var bitString = new StringBuilder();

            foreach (var token in tokens)
            {
                if (huffmanCodes.ContainsKey(token))
                {
                    bitString.Append(huffmanCodes[token]);
                }
            }

            return BitsToBytes(bitString.ToString());
        }

        public string DecodeCompressed(byte[] compressed)
        {
            string bitString = BytesToBits(compressed);
            var tokens = new List<int>();
            var currentNode = huffmanTree;

            foreach (char bit in bitString)
            {
                if (currentNode == null)
                    break;

                currentNode = bit == '0' ? currentNode.Left : currentNode.Right;

                if (currentNode != null && currentNode.IsLeaf)
                {
                    tokens.Add(currentNode.TokenId.Value);
                    currentNode = huffmanTree;
                }
            }

            return baseTokenizer.Decode(tokens);
        }

        private byte[] BitsToBytes(string bits)
        {
            int numBytes = (bits.Length + 7) / 8;
            var bytes = new byte[numBytes];

            for (int i = 0; i < bits.Length; i++)
            {
                if (bits[i] == '1')
                {
                    bytes[i / 8] |= (byte)(1 << (7 - (i % 8)));
                }
            }

            return bytes;
        }

        private string BytesToBits(byte[] bytes)
        {
            var bits = new StringBuilder();

            foreach (var b in bytes)
            {
                for (int i = 7; i >= 0; i--)
                {
                    bits.Append((b & (1 << i)) != 0 ? '1' : '0');
                }
            }

            return bits.ToString();
        }

        public byte[] CompressWithGzip(string text)
        {
            var tokens = baseTokenizer.Encode(text);
            var tokenBytes = new byte[tokens.Count * sizeof(int)];

            Buffer.BlockCopy(tokens.ToArray(), 0, tokenBytes, 0, tokenBytes.Length);

            using (var output = new MemoryStream())
            {
                using (var gzip = new GZipStream(output, CompressionLevel.Optimal))
                {
                    gzip.Write(tokenBytes, 0, tokenBytes.Length);
                }
                return output.ToArray();
            }
        }

        public string DecompressWithGzip(byte[] compressed)
        {
            using (var input = new MemoryStream(compressed))
            using (var gzip = new GZipStream(input, CompressionMode.Decompress))
            using (var output = new MemoryStream())
            {
                gzip.CopyTo(output);
                var tokenBytes = output.ToArray();
                var tokens = new int[tokenBytes.Length / sizeof(int)];
                Buffer.BlockCopy(tokenBytes, 0, tokens, 0, tokenBytes.Length);
                return baseTokenizer.Decode(tokens.ToList());
            }
        }

        public double GetCompressionRatio(string text)
        {
            var original = Encoding.UTF8.GetBytes(text);
            var compressed = EncodeCompressed(text);
            return (double)compressed.Length / original.Length;
        }

        public void Save(string path)
        {
            baseTokenizer.Save(path + ".vocab");

            using (var writer = new StreamWriter(path + ".huffman"))
            {
                foreach (var kvp in huffmanCodes)
                {
                    writer.WriteLine($"{kvp.Key}\t{kvp.Value}");
                }
            }
        }

        public void Load(string path)
        {
            baseTokenizer.Load(path + ".vocab");

            huffmanCodes.Clear();
            using (var reader = new StreamReader(path + ".huffman"))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var parts = line.Split('\t');
                    if (parts.Length == 2 && int.TryParse(parts[0], out int tokenId))
                    {
                        huffmanCodes[tokenId] = parts[1];
                    }
                }
            }

            RebuildHuffmanTree();
        }

        private void RebuildHuffmanTree()
        {
            huffmanTree = new HuffmanNode();

            foreach (var kvp in huffmanCodes)
            {
                var current = huffmanTree;
                foreach (char bit in kvp.Value)
                {
                    if (bit == '0')
                    {
                        if (current.Left == null)
                            current.Left = new HuffmanNode();
                        current = current.Left;
                    }
                    else
                    {
                        if (current.Right == null)
                            current.Right = new HuffmanNode();
                        current = current.Right;
                    }
                }
                current.TokenId = kvp.Key;
            }
        }
    }

    #endregion

    #region SentencePiece-style Tokenizer

    public class SentencePieceTokenizer
    {
        private readonly Dictionary<string, int> pieces;
        private readonly Dictionary<int, string> reversePieces;
        private readonly int vocabSize;
        private readonly char[] specialChars = { '▁' }; // Underscore represents space

        public SentencePieceTokenizer(int vocabSize = 8000)
        {
            this.vocabSize = vocabSize;
            this.pieces = new Dictionary<string, int>();
            this.reversePieces = new Dictionary<int, string>();
            InitializeSpecialTokens();
        }

        private void InitializeSpecialTokens()
        {
            var tokens = new[] { "<unk>", "<s>", "</s>" };
            for (int i = 0; i < tokens.Length; i++)
            {
                pieces[tokens[i]] = i;
                reversePieces[i] = tokens[i];
            }
        }

        public void Train(List<string> corpus)
        {
            // Preprocess: add space marker
            var processedCorpus = corpus.Select(text => "▁" + text.Replace(" ", "▁")).ToList();

            // Initialize with character pieces
            var charFreq = new Dictionary<string, int>();
            foreach (var text in processedCorpus)
            {
                foreach (char c in text)
                {
                    string piece = c.ToString();
                    if (!charFreq.ContainsKey(piece))
                        charFreq[piece] = 0;
                    charFreq[piece]++;
                }
            }

            int idx = pieces.Count;
            foreach (var kvp in charFreq.OrderByDescending(x => x.Value))
            {
                if (!pieces.ContainsKey(kvp.Key))
                {
                    pieces[kvp.Key] = idx;
                    reversePieces[idx] = kvp.Key;
                    idx++;
                }
            }

            // Learn subword pieces using unigram language model
            while (pieces.Count < vocabSize)
            {
                var candidates = GenerateCandidatePieces(processedCorpus);
                if (candidates.Count == 0)
                    break;

                var bestPiece = candidates.OrderByDescending(x => x.Value).First().Key;

                if (!pieces.ContainsKey(bestPiece))
                {
                    pieces[bestPiece] = idx;
                    reversePieces[idx] = bestPiece;
                    idx++;
                }
            }
        }

        private Dictionary<string, int> GenerateCandidatePieces(List<string> corpus)
        {
            var candidates = new Dictionary<string, int>();

            foreach (var text in corpus)
            {
                for (int len = 2; len <= Math.Min(text.Length, 20); len++)
                {
                    for (int start = 0; start <= text.Length - len; start++)
                    {
                        string piece = text.Substring(start, len);
                        if (!pieces.ContainsKey(piece))
                        {
                            if (!candidates.ContainsKey(piece))
                                candidates[piece] = 0;
                            candidates[piece]++;
                        }
                    }
                }
            }

            return candidates;
        }

        public List<int> Encode(string text)
        {
            text = "▁" + text.Replace(" ", "▁");
            var tokens = new List<int> { pieces["<s>"] };

            int pos = 0;
            while (pos < text.Length)
            {
                int bestLen = 0;
                int? bestId = null;

                // Find longest matching piece
                for (int len = Math.Min(text.Length - pos, 20); len > 0; len--)
                {
                    string candidate = text.Substring(pos, len);
                    if (pieces.ContainsKey(candidate))
                    {
                        bestLen = len;
                        bestId = pieces[candidate];
                        break;
                    }
                }

                if (bestId.HasValue)
                {
                    tokens.Add(bestId.Value);
                    pos += bestLen;
                }
                else
                {
                    tokens.Add(pieces["<unk>"]);
                    pos++;
                }
            }

            tokens.Add(pieces["</s>"]);
            return tokens;
        }

        public string Decode(List<int> tokenIds)
        {
            var pieces = new List<string>();

            foreach (var id in tokenIds)
            {
                if (id == this.pieces["<s>"] || id == this.pieces["</s>"])
                    continue;

                if (reversePieces.ContainsKey(id))
                {
                    pieces.Add(reversePieces[id]);
                }
            }

            return string.Join("", pieces).Replace("▁", " ").Trim();
        }
    }

    #endregion
}
