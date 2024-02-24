import time
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, 'is running...\n')

# Set Hyperparameters
embedding_size = 512
multi_head_att = 16 # Multi Head Attention katmanındaki head sayısı
num_encoder_layers = 6 # Encoderdaki EncoderStack sayısı
num_decoder_layers = 6 # Decoderdaki DecoderStack sayısı
forward_expansion = 4 # Transformer Encoder/Decoder içindeki ileri beslemeli ağdaki nöronların artış oranı
learning_rate = 3e-4 # Adam optimizer'ının learning rate değeri
context_len = 256 # Girdi dizisinin (maksimum) uzunluğu (Padding uygulanacak karakter sayisi)
dropout = 0.10 # Dropout katmanlarının atma oranı

# Load Dataset
df = pd.read_csv('C:/Users/mehmet/.spyder-py3/Natural Language Processing/Transformers/eng_-french.csv',
                 encoding='utf-8')
print(df.head(), '\n')

# set padding value
french_vocab = ['|']
english_vocab = ['|']
start_character = False # kelimeler '|' bu ile basliyor mu kontrolu
french_out = None

for i in range(len(df)):
    
    french_out = list(set(df['French'][i])) # benzersiz karakterleri bulalim
    english_out = list(set(df['English'][i])) # benzersiz karakterleri bulalim
    
    if '|' in french_out or '|' in english_out:
        start_character = True
    
    for character in french_out: # French'in benzersiz tum karakterleri buluyoruz
        if character not in french_vocab:
            french_vocab.append(character) 
    
    for character in english_out: # English'de ki benzersiz tum karakterleri buluyoruz
        if character not in english_vocab:
            english_vocab.append(character)

print('French Set Variable: ', french_out)
print('Start Character: ', start_character, '\n')
print('French Vocab: ', french_vocab, '\n', len(french_vocab))
print()
print('English Vocab: ', english_vocab, '\n', len(english_vocab))

french_vocab = sorted(french_vocab)
english_vocab = sorted(english_vocab)
print('\nSorted English Vocab', english_vocab, '\n', len(english_vocab), '\n')

# stoi -> string to index
# itos -> index to string

fr_stoi = {character:index for index,character in enumerate(french_vocab)}
fr_itos = {index:character for index, character in enumerate(french_vocab)}
print('French String to Index: ', fr_stoi)

en_stoi = {character:index for index, character in enumerate(english_vocab)}
en_itos = {index:character for index, character in enumerate(english_vocab)}
print('\nEnglish Index to String: ', en_itos, '\n')


# =============================================================================
# Train data setindeki farkli cumlelerin uzunluklari degiskendir. Ancak, bir
# egitim batch'inde ki tum cumlelerin ayni boyutta olmasi gerekir.
# Dolayisiyla, farkli uzunluktaki cumleleri ayni boyuta getirmek icin padding tokenleri
# kullaniriz. 
# Bu tokenler daha sonra modelin islemesi gerekmeyen padding bilgisini temsil eder.
# Bu nedenle, train sirasinda modele bu tokenlerin dikkate alinmamasi gerektigi belirtilir.
# Padding tokenlerinin indexleri 'stoi(string to index)' dict'lerinden elde edilir.
# Bu indexler, dolgu tokenlerine karsilik gelen indexlerdir ve train sirasinda kullanilir.
# =============================================================================

PADDING_fr = fr_stoi['|'] # padding tokeninin indexi
PADDING_en = en_stoi['|'] # padding tokeninin indexi
print('PADDING_fr: ', PADDING_fr)
print('PADDING_en: ', PADDING_en)

fr_vocab_size = len(french_vocab) # Benzersiz kelime haznesi boyutu
en_vocab_size = len(english_vocab) # Tokenizer tarafindan bilinen token sayisi


def tokenizer(string, stoi):
    
    result = []
    for character in string:
        result.append(stoi[character])
    
    if len(result) < context_len:
        result += [0]*(context_len - len(result)) # string context_len uzunluguna erisene kadar sonuna 0 ekle
    
    return torch.tensor(result, dtype=torch.long)

print('\n', tokenizer("Bears never dance at the circus", en_stoi)[:50])


def decode_tokenizer(indexs, itos):
    
    result = ""
    for idx in indexs:
        result += itos[idx.item()]

    return result

print(decode_tokenizer(tokenizer("Bears never dance at the circus", en_stoi), en_itos))


encode_english = []
for text in df['English']:
    encode_english.append(tokenizer(text, en_stoi))

print('\nTokenize English Text:\n', encode_english[0][: 20])
print('\nDeTokenize English Text:\n', decode_tokenizer(encode_english[0], en_itos))


encode_french = []
for text in df['French']:
    encode_french.append(tokenizer(text, fr_stoi))

print('\nTokenize French Text:\n', encode_french[0][: 20])
print('\nDeTokenize French Text:\n', decode_tokenizer(encode_french[0], fr_itos))



# =============================================================================
# Creating Transformers Block

    # 1. Embedding Block:
        # Bu blok, girdi verilerini bir vektor temsiline donusturmek icin kullanilicak.
        # Bu blok 2 yapidan olusur. Bunlar:
            # 1.1 Token Embedding
            # 1.2 Positional Encoding
# =============================================================================

# =============================================================================
# 1.1 Token Embedding:
# =============================================================================


class TokenEmbeddingLayer(nn.Module):
    """
    Bu sinif bir token embedding layer olusturur. Token Embedding Layer,
    bir kelime dagarcigini belirli bir boyuttaki bir vektor uzatina donusturur.
    Bu islem, bir kelimenin bir vektorle temsil edilmesini saglar ve ML modelleri
    icin giris olarak kullanilabilir.
    """
    
    def __init__(self, embedding_size, vocab_size, context_len):
        """
        Initialize Method

        Parameters
        ----------
        embedding_size : int
            Embedding boyutu. Her kelimenin embedding vektorunun boyutunu belirleyen bir sayi.
        vocab_size : int
            Vocabulary size. Kelime dagarcinin boyutunu belirten bir sayi.
        context_len : int
            Context length. Giris dizisinin maksimum uzunlugunu belirten bir sayi.
        """
        super(TokenEmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
    
    def forward(self, x):
        """
        Forward Method
        
        Modelin girisini alir ve ileri dogru gecis yaparak ciktiyi hesaplar.
        Forward islemi "nn.Embedding" katmanini kullanarak giris dizisini bir
        Embedding matrisine donusturur.
        Bu Embedding matrisi, her bir giris belirtecinin karsilik gelen
        Embedding Vektorunu icerir.

        Parameters
        ----------
        x : Tensor
            Giris verisi. Herbir oge, kelime veya tokenin bir sayisal temsilidir.
            Ornegin, "Merhaba, Dunya!". Merhaba 10. index, Dunya 20. index ve
            ! 30. index olsun.
            Bu durumda x -> [10, 20, 30]

        Returns
        -------
        out : Tensor
            Embedding katmanini sonucu elde edilen cikis.
        """
        out = self.token_embedding(x)
        return out


# Example Embedding Layer
context_len_ = 256
embedding_size_ = 512
embedding_layer = TokenEmbeddingLayer(embedding_size_,
                                      en_vocab_size,
                                      context_len_)
print('\nEmbedding Layer: ', embedding_layer)

embeddings = embedding_layer(tokenizer('Bears never dance at the circus',
                                       en_stoi))
print('\nEmbeddings: ', embeddings)
print('\nEmbedding Shape: ', embeddings.shape)



# =============================================================================
# 1.2 Positional Encoding:
# =============================================================================

# d_model = embedding_size
# PE = Positional Encoding

# PE(pos,2i)=sin(pos/10000^(2i/dmodel)) 
# PE(pos,2i+1)=cos(pos/10000^(2i/dmodel))


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    
    Embedding islemine konumsal kodlama eklemek icin kullanilir.
    Bu class, her bir giris pozisyonu icin bir konumsal kod uretir.
    Bu kodlar, her bir pozisyon icin benzersiz bir desen olusturur ve modelin giris
    dizisindeki her bir elemanin konumunu temsil etmesine yardimci olur.
    """
    def __init__(self, embedding_size, max_sequence_length):
        """
        Initialize Method

        Parameters
        ----------
        embedding_size : int
            Embedding dimension.
        max_sequence_length : int
            Girdi dizisinin mx uzunlugu.
        """
        super(PositionalEncoding, self).__init__()
        self.max_sequence_length = max_sequence_length # En uzun girdi dizisinin uzunlugu
        self.embedding_size = embedding_size # Embedding boyutu
    
    def forward(self):
        """
        Forward Method
        
        Ilk olarak, her pozisyonun bir araligini ve bu aralik icin sinus ve kosinus
        terimlerini hesaplamalari yapar.
        Ardindan, bu hesaplamalari kullanarak pozisyonel kodlamayi olusturur
        ve dondurur.
        """
        
        # position; her pozisyonun 0'dan baslayarak max dizi uzunluguna kadar olan sayisal bir dizi olusturur.
        # Bu pozisyonlar, her bir ogenin sirasini temsil eder.
        position = torch.arange(self.max_sequence_length, device=device,
                                dtype=torch.float32).unsqueeze(1)
        print('\nPosition: ', position)
        
        # div_term; her boyut icin pozisyonlara gore dizi uzunlugu boyunca esit aralikli terimler olusturulur.
        # Bu terimler, her bir boyutun dalgaboyunu temsil eder ve dalgaboyu boyutundaki bir degisiklinin
        # etkisini kontrol eder.
        div_term = torch.exp(torch.arange(0, self.embedding_size, 2, device=device,
                                          dtype=torch.float32) * (-math.log(10000.0) /self.embedding_size))
        print('\nDiv Term: ', div_term, '\n', len(div_term))

        # pe; pozisyonel kodlamayi tutmak icin bir tensor olusturur. Bu tensor, max dizi uzunlugu
        # ve belirlenen embedding boyutu boyunca sifirlardan olusur.
        pe = torch.zeros(self.max_sequence_length, self.embedding_size, device=device, dtype=torch.float32)
        print('\nPE: ', pe)
        
        # pe; pozisyonel kodlama tensorundeki cift indexli boyutlara, sinusoidal terimlerin uygulanmasini
        # saglar. Bu, pozisyonlarin cift indexli boyutlarda sinus fonk. kodlanmasini temsil eder.
        pe[:, 0::2] = torch.sin(position * div_term)
        print('\nPE[:, 0::2]: ', pe[:, 0::2]) # pe[:, 1::2] tek indexliler burada 0 olarak kaliyor.
        
        # pe; pozisyonel kodlama tensorundeki tek indexli boyutlara, kosinus terimlerinin uygulanmasini
        # saglar. Bu pozisyonlarin tek indexli boyutlarda kosinus fonk. ile kodlanmasini temsil eder.
        pe[:, 1::2] = torch.cos(position * div_term)
        print('\nPE[:, 1::2]: ', pe[:, 1::2]) # tek indexlilerde cos ile degerlerini aliyor.
        
        return pe

# =============================================================================
# 1. Embedding Block

    # Embedding Block, girdi verilerini bir vektor temsiline donusturmek icin kullanikacak.
    # Bu blok 2 kisimdan olusuyordu. Bunlar;
        # 1.1 Token Embedding,
        # 1.2 Positional Encoding
    # Simdi yukarida olusturdugumuz iki bloguda tek bir yerde birlestirelim.
    # Bu nedenle, bu iki blogu birlestiren bir kapsayici class olusturalim.
# =============================================================================

print('\n\n-----------------Embedding Block-----------------\n\n')


class EmbeddingLayer(nn.Module):
    """
    Embedding Layer
    
    Token Embedding ve Positional Encoding islemierini icerir.
    """
    def __init__(self, embedding_size, vocab_size, block_size):
        """
        Initialize Method

        Parameters
        ----------
        embedding_size : int
            Embedding dimension.
        vocab_size : int
            Kelime haznesinin boyutu. Kullanilan dildeki toplam benzersiz kelime sayisi.
        block_size : int
            Blok boyutunu belirtir. Bu, girdi dizisinin max uzunlugunu belirler.
        """
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = TokenEmbeddingLayer(embedding_size, vocab_size, block_size)
        self.positional_encoding = PositionalEncoding(embedding_size, block_size)
    
    def forward(self, x):
        """
        Forward Method
        
        Sinifin ileri gecis yontemidir ve sinifin bir ornegi uzerinde cagrildiginda calisir.
        Ilk olarak, girdi dizisinin her bir ogesini Token Embedding Layer'dan gecirir.
        Ardindan, Positional Encoding Layer'dan elde edilen pozisyonel kodlamayi alir.
        Son olarak, Token Embedding ve Positional Encoding ciktilarini toplar ve dondurur.

        Parameters
        ----------
        x : array or tensor
            Girdi dizisi. Ornegin, cumlelerin tokenize edilmis bir listesi.

        Returns
        -------
        out : array or tensor
            Girdi dizisinin her bir elemaninin Embedding ve Positional Encoding islemlerinden
            gecirilmesiyle el edilen dizi veya tensor.
        """
        out = self.token_embedding(x).to(device) + self.positional_encoding()
        return out


# Example Embedding Layer
context_len_ = 256
embedding_size_ = 512
embedding_layer = EmbeddingLayer(embedding_size_, en_vocab_size, context_len_)
embeddings = embedding_layer(tokenizer('Bears never dance at the circus', en_stoi))
print('\n\n-----------------Embedding Block-----------------\n\n')
print('Embeddings Shape: ', embeddings.shape)



# =============================================================================
# Key, Value, Query Matrixi Olusturalim
# =============================================================================

questions = ['Who', 'When', 'Where', 'What', 'How Many']
sentence = 'Bears never dance at the circus'
tokens = sentence.split(' ')

# embedding_Size = d_model(makalede embed size) = 5
# context_len = 6

# T ve C, attention mekanizmasiyla hesaplanan agirliklarin do*gru sekilde hesaplanmasi
# ve uygun bir sekilde normalize edilmesidir.
T, C = (len(tokens), 5) # context length, tokens length
print('\nT: ', T)
print('C: ', C)

# query, her bir tokenin(kelimenin) dikkat dagilimini belirlemek icin kullanilir. Bir cumlenin
# tokenlerinin embedding temsillerinden olusur ve *dikkat dagilimini belirlemek* icin kullanilir.
query = [
    [2,2,7,6,3], # Bears
    [3,5,3,5,1], # never
    [7,4,5,5,2], # dance
    [7,3,8,4,2], # at
    [6,1,5,4,2], # the
    [7,4,5,7,2], # circus
    ]
print('\nQuery Matrix:\n', query)

# key, her bir tokenin temsilini belirlemek icin kullanilir. Cumlenin tokenlerinin embedding
# temsillerinden olusur ve *dikkat agirliklarini hesaplamak* icin kullanilir.
key = [
    [9,1,1,7,8], # Bears
    [1,6,1,1,1], # never
    [3,7,5,6,3], # dance
    [4,8,5,5,3], # at
    [1,1,1,3,4], # the
    [3,7,9,8,4], # circus
    ]
print('\nKey Matrix:\n', key)

# value, her bir tokenin temsilini belirlemek icin kullanilir. Cumlenin tokenlerinin embedding
# temsillerinden olusur ve *dikkat mekanizmasinin ciktisini olusturmak* icin kullanilir.
value = [
    [9,1,1,7,8], # Bears
    [1,6,1,1,1], # never
    [3,7,5,6,3], # dance
    [4,8,5,5,3], # at
    [1,1,1,3,4], # the
    [3,7,9,8,4], # circus
    ]
print('\nValue Matrix:\n', value)


# numpy array -> torch tensor
# 0-1 arasina normalize ediyoruz (10'a boluyoruz)
query = torch.tensor(query) / 10
key = torch.tensor(key) / 10
value = torch.tensor(value) / 10


# =============================================================================
# Softmax Fonksiyon ve Matris Transpozunu hatirlayalim
# =============================================================================
# Softmax Function -> np.exp(x) / np.sum(np.exp(x))  ->  [0,1]
# Transpose -> A = [[1,2],[3,4]]2x2    A.T = [[1,3],[2,4]]2x2
# Transpose -> B = [[1,3,2],[3,0,4]]2x3    B.T = [[1,3],[2,0],[2,4]]3x2
# 1   3   2       1   3
# 3   0   4   ->  3   0
#                 2   4


# Attention Mechanism, bir modelin bir girdi dizisindeki her bir elemana ne kadar dikkat
# etmesi gerektigini belirler. Bu durum, ozellikle ceviri, metin siniflandirma ve benzeri
# NLP gorevlerinde faydalidir.
# Bu islemde, Query ve Key matrixlerinin Transpozunu(key.T) carparak ic carpim alinir.
# Daha sonra bu carpim, kok C'ye bolunerek(C**0.5) softmax fonk. verilir.
# Burada C, query ve key matrixlerinin boyutunu temsil eder. 
# Kok C'ye bolunme islemi, softmax sonuclarinin genel olarak attention agirliklarini
# stabilize etmeye ve olceklemeye yardimci olur.
# Sonuc olarak; bu islem, Attention Weights hesaplanmasini saglar. Daha sonra bu agirliklar,
# Value matrisi ile carpilarak Attention Vector elde edilir.
# Bu Vector, modelin onemli girdi ogelerine daha fazla dikkat gostermesini saglar.
att = F.softmax((query @ key.T) / (C ** 0.5), dim=1) # @ -> pytorch'da matrix carpimini ifade eder

print('********************* Q K V MATRIX *********************')

print('\nQuery Matrix:\n', query)
print('\nKey Matrix:\n', key)
print('\nAttention Matrix:\n', att)



# =============================================================================
# 2.1 Self Attention (Single Head)
# =============================================================================


class SingleHeadAttention(nn.Module):
    """
    Single Head Attention
    
    Giris attention agirliklarini hesaplamak icin tek bir attention head kullanir.
    """
    
    def __init__(self, embedding_size, head_size):
        """
        Initialize Method

        Parameters
        ----------
        embedding_size : int
            Embedding dimension.
        head_size : int
            Her bir attention head'in boyutu.
        """
        super(SingleHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.query = nn.Linear(embedding_size, head_size, bias=False) # weight matrix
        self.key = nn.Linear(embedding_size, head_size, bias=False) # weight matrix
        self.value = nn.Linear(embedding_size, head_size, bias=False) # weight matrix
    
    def forward(self, x):
        """
        Forward Method

        Parameters
        ----------
        x : array or tensor
            Modele gelen giris verisi.

        Returns
        -------
        context_vector : tensor
            Dikkat Mekanizmasinin cikisi.
        """
        B,T,C = x.shape # B->Batch Size, T->x'in max length, C->Ozellik Vektoru boyutu
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        att = F.softmax((query @ key.T) / (self.head_size ** 0.5), dim=1)
        context_vector = att @ value # Attention Mechanism Output (Attention Vector)
        
        return context_vector



# =============================================================================
# 2.2 Multi Head Attention

    # h adet dikkat basligina sahiptir.
    # Her dikkat basliginin ayri bir query, key ve value matrixi vardir.
# =============================================================================



class MultiHeadAttention(nn.Module):
    
    def __init__(self, embedding_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        
        assert embedding_size % num_heads == 0, 'embedding_size, num_heads\'e bolunebilmelidir.'
        
        # Her basliga, embedding_size'i num_heads'e bolunmus boyutunu atamak icin;
        self.dimension_qkv = embedding_size // num_heads
        
        # Lineer Projeksiyon, bir vektoru/matrixi baska bir boyuta donusturmek icin kullanilir.
        
        # Q, K, V projekte etmek icin nn.Linear kullaniyoruz. Ayni lineer projeksionu tum basliklara
        # uyguluyoruz. Bunun nedeni; Q,K ve V'leri projekte etmek icin ayri ayri matrixler
        # kullanmak yerine tek bir matrix carpimi kullanmamiza olanak tanimasidir.
        # Bu, her biri icin ayri matrixler kullanmaktan daha verimlidir(memory).
        self.W_queries = nn.Linear(embedding_size, embedding_size, bias=False)
        self.W_keys = nn.Linear(embedding_size, embedding_size, bias=False)
        self.W_values = nn.Linear(embedding_size, embedding_size, bias=False)
        
        # Ciktiyi projekte etmek icin tek bir Lineer Projeksiyon kullaniyoruz.
        self.lineer_proj = nn.Linear(embedding_size, embedding_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query_src, key_src, value_src, mask=None):
        # Girdi yigininin seklini alalim. Cunku, modelin her bir boyuttaki girdileri nasil
        # isleyecegini ve ciktilari nasil olusturacagini belirler.
        B,T,C = key_src.shape # B->Batch Size, T->Sequent Length, C->Ebbedding Size
        
        # Q, K ve V'leri ilgili agirlik matrixlerini kullanarak projekte edelim
        queries = self.W_queries(query_src) # batch_size, seq_len, embedding_size
        keys = self.W_keys(key_src) # batch_size, seq_len, embedding_size
        values = self.W_values(value_src) # batch_size, seq_len, embedding_size
        
        # Q, K ve V'leri coklu basliklara bolmek icin yeniden sekillendiriyoruz
        queries = queries.view(B, T, self.num_heads, self.dimension_qkv) # batch_size, seq_len, num_heads, dimension_qkv
        keys = keys.view(B, T, self.num_heads, self.dimension_qkv) # batch_size, seq_len, num_heads, dimension_qkv
        values = values.view(B, T, self.num_heads, self.dimension_qkv) # batch_size, seq_len, num_heads, dimension_qkv
        
        # Q, K ve V'leri tensorun seklini (batch_size, num_heads, seq_len, dimension_qkv)
        # yapacak sekilde yer degistirelim
        queries = queries.transpose(1, 2) # batch_size, num_heads, seq_len, dimension_qkv
        keys = keys.transpose(1, 2) # batch_size, num_heads, seq_len, dimension_qkv
        values = values.transpose(1, 2) # batch_size, num_heads, seq_len, dimension_qkv
        
        # Attention puanlari hesapliyoruz
        attn_score = queries @ keys.transpose(-2, -1) # batch_size, num_heads, seq_len, seq_len
        
        # Attention puanlarini olceklendirip maskeyi uyguluyoruz (varsa)
        scaled_attn_score = attn_score / self.dimension_qkv ** -0.5
        if mask is not None:
            scaled_attn_score = scaled_attn_score.masked_fill(mask==0, float('-inf'))
        
        # Attention agirliklarini hesaplamak icin Softmax Aktivasyon Fonk. uygulayalim
        attention_weights = torch.softmax(scaled_attn_score, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Attention agirliklarini V ile carpalim
        out = attention_weights @ values
        out = out.transpose(1, 2) # batch_size, seq_len, num_heads, dimension_qkv
        
        # Matrixi (batch_size, seq_len, embedding_size) sekline donusturmak icin yeniden reshape yapalim
        out = out.reshape(B, T, C) # batch_size, seq_len, embedding_size
        
        # bir sonraki katmana beslemek icin son bir lineer projeksiyon uygulayalim
        out = self.lineer_proj(out)
        
        return out


# =============================================================================
# 3. Feed Forward Network (Ileri Beslemeli Sinir Agi) - FFN

    # Bu agin icinde, 2 Lineer Katman ve bu katmanlar arasinda ReLU Akt. Func. var.
    # Agir input ve output boyutu ayni kalir ancak iceride boyutu bir katsayi ile
    # carpmak uzere artırırız.
    # Bu katsayiya "Forward Expansion" denir.
# =============================================================================


class FeedForwardNet(nn.Module):
    
    def __init__(self, embedding_size, forward_expansion, dropout=0.1):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(embedding_size, int(embedding_size * forward_expansion))
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(int(embedding_size * forward_expansion), embedding_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out



# =============================================================================
# 4. Encoder Stack

    # Encoder Stack;
        # 1 tane Multi Head Attention ve
        # 1 tane Feed Forward Network blogundan olusur.
    # Bu blogun her alt birimini Layer Normalization ve Residual Connection takip eder.
# =============================================================================


class EncoderStack(nn.Module):
    
    def __init__(self, embedding_size, num_heads, forward_expansion, dropout=0.1):
        super(EncoderStack, self).__init__()
        
        self.MHA = MultiHeadAttention(embedding_size, num_heads, dropout)
        self.FFN = FeedForwardNet(embedding_size, forward_expansion, dropout)
        
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        out = x + self.dropout1(self.MHA(x, x, x)) # Dropout + Residual Connection
        norm_out = self.layer_norm1(out)
        out = norm_out + self.dropout2(self.FFN(norm_out)) # Dropout + Residual Connection
        norm_out = self.layer_norm2(out)
        
        return norm_out



# =============================================================================
# 5. Encoder

    # Encoder blogu, bir Embedding Katmani ve n adet Encoder Stack
    # blogundan (n=num_layers) olusur.
# =============================================================================


class Encoder(nn.Module):
    
    def __init__(self, vocab_size, block_size, embedding_size, num_heads, forward_expansion, num_layers):
        super(Encoder, self).__init__()
        self.block_size = block_size
        self.embedding_size = embedding_size
        
        self.embedding_layer = EmbeddingLayer(embedding_size, vocab_size, block_size)
        
        self.layers = nn.ModuleList([EncoderStack(embedding_size, num_heads, forward_expansion)
                                     for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.embedding_layer(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x



# =============================================================================
# 6. Decoder Stack

    # Decoder Stack blogu, 2 Multi Head Attention (1'i maskelemeli attention,
    # digeri capraz attention icin) ve 1 Feed Forward Network blogundan olusur.
    # Bu blogunda her alt birimi Layer Normalization ve Residual Connection takip eder.
# =============================================================================


class DecoderStack(nn.Module):
    
    def __init__(self, embedding_size, num_heads, forward_expansion, dropout=0.1):
        super(DecoderStack, self).__init__()
        
        self.masked_MHA = MultiHeadAttention(embedding_size, num_heads, dropout=dropout)
        self.crossed_MHA = MultiHeadAttention(embedding_size, num_heads, dropout=dropout)
        
        self.FFN = FeedForwardNet(embedding_size, forward_expansion, dropout=dropout)
        
        self.layerNorm1 = nn.LayerNorm(embedding_size)
        self.layerNorm2 = nn.LayerNorm(embedding_size)
        self.layerNorm3 = nn.LayerNorm(embedding_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_out, target_mask):
        
        masked_attn_out = self.dropout1(self.masked_MHA(query_src=x, key_src=x,
                                                        value_src=x,
                                                        mask=target_mask))
        masked_attn_out = self.layerNorm1(masked_attn_out + x)
        
        crossed_attn_out = self.dropout2(self.crossed_MHA(query_src=masked_attn_out,
                                                          key_src=encoder_out,
                                                          value_src=encoder_out))
        crossed_attn_out = self.layerNorm2(crossed_attn_out + masked_attn_out)
        
        ffn_out = self.dropout3(self.FFN(crossed_attn_out))
        ffn_out = self.layerNorm3(ffn_out + crossed_attn_out)
        
        return ffn_out



# =============================================================================
# 7. Decoder

    # Decoder Blogu, bir Embedding Katmani ve n adet Decoder Stack Blogundan
    #(n=num_layers) olusur.
# =============================================================================


class Decoder(nn.Module):
    
    def __init__(self, vocab_size, context_len, embedding_size, num_heads, forward_expansion, num_layers):
        super(Decoder, self).__init__()
        self.context_len = context_len
        self.embedding_size = embedding_size
        
        self.embedding_layer = EmbeddingLayer(embedding_size, vocab_size, context_len)
        
        self.layers = nn.ModuleList([DecoderStack(embedding_size, num_heads, forward_expansion)
                                     for _ in range(num_layers)])
    
    def forward(self, x, encoder_output, target_mask):
        x = self.embedding_layer(x)
        
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask)
        
        return x



# =============================================================================
# 8. Transformer

    # Transformer modeli, Encoder, Decoder bloklarinin birlesiminden olusur.
    # Encoder, Embedding katmani ve n adet Encoder Stack katmanindan olusur.
    # Decoder ise Embedding katmani ve n adet Decoder Stack katmanindan olusur.
    # En son da bu modelin sonuna cikti boyutu Tokenizer'in ogrendigi kelime
    # sayisina esit bir lineer katman koyariz.
    # Bu son katman sayesinde Decoder'dan aldigimiz ciktiyi tahmin yapabilmek
    # icin kullaniabilecek hale getirmis oluruz.
# =============================================================================


class Transformer(nn.Module):
    
    def __init__(self, en_vocab_size, fr_vocab_size, block_size, embedding_size,
                 num_heads, num_encoder_layers, num_decoder_layers,
                 forward_expansion, learning_rate, dropout=0.1):
        
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(vocab_size=en_vocab_size, block_size=block_size,
                               embedding_size=embedding_size, num_heads=num_heads,
                               forward_expansion=forward_expansion, num_layers=num_encoder_layers)
        
        self.decoder = Decoder(vocab_size=fr_vocab_size, context_len=block_size,
                               embedding_size=embedding_size, num_heads=num_heads,
                               forward_expansion=forward_expansion, num_layers=num_decoder_layers)
        
        self.embedding_size = embedding_size
        self.linear = nn.Linear(embedding_size, fr_vocab_size)
    
    def forward(self, source, target, source_mask=None, target_mask=None):
        
        encoder_output = self.encoder(source)
        decoder_output = self.decoder(target, encoder_output, target_mask()) # batch_size, context_len, embedding_size
        
        output = self.linear(decoder_output) # batch_size, context_len, fr_vocab_size
        
        return output



en_vocab_size = 12000
fr_vocab_size = 12000
context_len = 512
embedding_size = 1024
num_heads = 16
num_encoder_layers = 6
num_decoder_layers = 6
forward_expansion = 4
learning_rate = 1e-4
dropout = 0.3

transformer = Transformer(en_vocab_size=en_vocab_size, fr_vocab_size=fr_vocab_size,
                          block_size=context_len, embedding_size=embedding_size,
                          num_heads=num_heads, num_encoder_layers=num_encoder_layers,
                          num_decoder_layers=num_decoder_layers, forward_expansion=forward_expansion,
                          learning_rate=learning_rate, dropout=dropout).to(device)

print('\nResult:\n', (sum(p.numel() for p in transformer.parameters())) / 10 ** 6, 'Million Parameters')

end_time = time.time()
print('\nElapsed Time: ', (end_time - start_time))
