#coding:utf-8
import math
import re
import time
import copy
import os
import pickle


class Timer:
    """calculate the time"""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self,*args):
        self.end = time.time()
        self.interval = self.end - self.start


class Syllable(object):
    """411个音节"""
    def __init__(self,filename):
        self.syllables = set()

        filepath = os.path.abspath(".") + os.sep + filename
        with open(filepath) as f:
            print "Loading pinyin syllable list..."
            for line in f:
                line = line.strip('\n')
                self.syllables.add(line)
            print "Done"
        pass

    def IsValidSyllable(self,syllable):
        if syllable in self.syllables:
            return True
        else:
            return False


class Lexicon(object):
    """词典"""
    class Node(object):
        def __init__(self):
            self.children = dict()
            self.phrases = list()


    def __init__(self,filename,syllables):
        self.root = self.Node()
        self.max_length = 0
        self.syllables = syllables
        self.lexicon = set()

        filepath = os.path.abspath(".") + os.sep + filename
        with open(filepath) as f:
            print "Loading dictionary..."
            for line in f:
                line = line.strip('\n')
                self.ParseLine(line)
            print "Done"

    def ParseLine(self,line):
        [s,phrase] = line.strip('\n').split('\t')
        section = s.split('\'')
        self.InsertPhrase(self.root,phrase,section,0)

    def InsertPhrase(self,current,phrase,pinyin_section,index):
        pinyin = pinyin_section[index]

        if not self.syllables.IsValidSyllable(pinyin):
            return

        if not current.children.has_key(pinyin):
            child = self.Node()
            current.children[pinyin] = child
        else:
            child = current.children[pinyin]

        if index == len(pinyin_section) - 1:
            child.phrases.append(phrase)
            self.lexicon.add(phrase)
            if len(phrase)/3 > self.max_length:
                self.max_length = len(phrase)/3
        else:
            self.InsertPhrase(child,phrase,pinyin_section,index+1)



    def GetPhrases(self,pinyin_sequence):
        return self.GetPhrasesAbstract(self.root,pinyin_sequence,0)


    def GetPhrasesAbstract(self,current,pinyin_section,index):
        pinyin = pinyin_section[index]

        if not current.children.has_key(pinyin):
            return None

        child = current.children[pinyin]

        if index == len(pinyin_section) - 1:
            return child.phrases
        else:
            return self.GetPhrasesAbstract(child,pinyin_section,index+1)


    def Contains(self,phrase):
        if phrase in self.lexicon:
            return True
        return False


class LanguageModel(object):
    unknown = '<unknown>'
    inifnitesmial = pow(math.e,-200)

    def __init__(self,filename,lexicon):
        print "Loading statistical LanguageModel ..." 
        self.lexicon = lexicon
        self.stage = 0
        self.unigram = dict()
        self.bigram = dict()

        filepath = os.path.abspath(".") + os.sep + filename
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                self.ParseLine(line)

        print "Done"

    def ParseLine(self,line):
        if self.stage == 0:
            if line.startswith("\\1-gram\\"):
                self.stage += 1
        elif self.stage == 1:
            if line.startswith("\\2-gram\\"):
                self.stage += 1
            else:
               result = re.split("\s+",line)
               phrase = result[0]
               probability = float(result[1])
               if not self.lexicon.Contains(phrase):
                   return False

               self.unigram[phrase] = probability
        elif self.stage == 2:
            result = re.split("\s+",line)
            phrase1 = result[0]
            phrase2 = result[1]
            probability = float(result[2])
            
            if phrase1 != self.unknown and not self.lexicon.Contains(phrase1):
                return False
            if phrase2 != self.unknown and not self.lexicon.Contains(phrase1):
                return False
        
            if not self.bigram.has_key(phrase1):
                dic = dict()
                self.bigram[phrase1] = dic
            else:
                dic = self.bigram[phrase1]

            dic[phrase2] = probability

        return False

    def GetUnigram(self,phrase):
        if self.unigram.has_key(phrase):
            return self.unigram[phrase]
        return self.inifnitesmial

    def GetBigram(self,phrase1,phrase2):
        delta = 0
        if(self.bigram.has_key(phrase1)):
            dic = self.bigram[phrase1]
            if dic.has_key(phrase2):
                return dic[phrase2]
            elif dic.has_key(self.unknown):
                delta = dic[self.unknown]
        elif self.bigram.has_key(self.unknown):
            dic = self.bigram[self.unknown]
            if dic.has_key(phrase2):
                delta = dic[phrase2]
        return self.GetUnigram(phrase1)*self.GetUnigram(phrase2)*(math.e + delta)


class Graph(object):
    class Vertex(object):
        def __init__(self,ID):
            self.f = list()    #"f" means "from" ,for "from" is one of python's keyword
            self.to = list()
            self.data = None
            self.ID = ID

    class Edge(object):
        def __init__(self,f,to):
            self.f = f
            self.to = to
            self.data = None

    def __init__(self):
        self.vertex = list()
        self.edges = list()
        self.vertex_count = 0
        self.edge_count = 0

    def InitializeVertex(self):
        for i in range(self.vertex_count):
            self.vertex.append(self.Vertex(i))

    def AddEdge(self,f,to):
        e = self.Edge(f,to)
        f.to.append(e)
        to.f.append(e)
        self.edge_count += 1
        self.edges.append(e)
        return e 

class SyllableGraph(Graph):
    pinyin_max_length = 6

    class VertexData(object):
        def __init__(self):
            self.accessed_forward = False
            self.accessed_backward = False
            self.shrinked_id = -1

    def __init__(self,string,syllables):
        print "Making SyllableGraph ..."
        super(SyllableGraph,self).__init__()
        self.pinyin_string = string
        self.valid_syllables = syllables
        self.pinyin_sequence = list()
        self.pinyin_sequences = list()
        self.MakeGraph();
        print "Done"

    def MakeGraph(self):
        self.vertex_count = len(self.pinyin_string) + 1
        self.InitializeVertex()
        for i in range(self.vertex_count):
            self.vertex[i].data = self.VertexData()

        for i in range(len(self.pinyin_string)):
            j = 1
            while j <= self.pinyin_max_length and (i+j) <= len(self.pinyin_string):
                pyslice = self.pinyin_string[i:i+j]
                if self.valid_syllables.IsValidSyllable(pyslice):
                    e = self.AddEdge(self.vertex[i],self.vertex[i+j])
                    e.data = pyslice
                j += 1
        self.ShrinkGraph()

    def ShrinkGraph(self):
        self.SearchForward(self.vertex[0])

        venddata = self.vertex[self.vertex_count-1].data
        if not venddata.accessed_forward:
            raise Exception("InvalidSyllable")
        
        self.SearchBackward(self.vertex[self.vertex_count-1])

        valid_vertex_count = 0

        for i in range(self.vertex_count):
            vdata = self.vertex[i].data
            if vdata.accessed_forward and vdata.accessed_backward:
                vdata.shrinked_id = valid_vertex_count
                valid_vertex_count += 1

        shrinked = list()
        for i in range(valid_vertex_count):
            shrinked.append(self.Vertex(i))
            shrinked[i].data = self.VertexData()
            shrinked[i].data.shrinked_id = i;

        
        for i in range(self.vertex_count):
            vdata = self.vertex[i].data
            if vdata.shrinked_id >= 0:
                for e in self.vertex[i].to:
                    evdata = e.to.data
                    if evdata.shrinked_id >= 0:
                        e_new  = self.AddEdge(shrinked[vdata.shrinked_id],shrinked[evdata.shrinked_id])
                        e_new.data = e.data

        self.vertex = shrinked
        self.vertex_count = valid_vertex_count

    def SearchForward(self,v):
        vdata = v.data
        vdata.accessed_forward = True
        for e in v.to:
            evdata = e.to.data
            if not evdata.accessed_forward:
                self.SearchForward(e.to)

    def SearchBackward(self,v):
        vdata = v.data
        vdata.accessed_backward = True
        for e in v.f:
            evdata = e.f.data
            if not evdata.accessed_backward:
                self.SearchBackward(e.f)
    
    def GetPinYinSequences(self,max_length):
        for i in range(self.vertex_count):
            self.MakePinyinSequence(self.vertex[i],max_length,i)

        return self.pinyin_sequences

    def MakePinyinSequence(self,v,limit,f):
        for e in v.to:
            self.pinyin_sequence.append(e.data)
            self.pinyin_sequences.append(PinYinSequence(copy.deepcopy(self.pinyin_sequence),f,e.to.data.shrinked_id))

            if limit > 1:
                self.MakePinyinSequence(e.to,limit-1,f)

            print self.pinyin_sequence
            self.pinyin_sequence.pop(len(self.pinyin_sequence)- 1)
            

class PinYinSequence(object):
    def __init__(self,pinyin_sequence,f,to):
        self.pinyin_sequence = pinyin_sequence
        self.f = f
        self.to = to

    def Pinyins(self):
        return self.pinyin_sequence


class LexiconGraph(Graph):
    class EdgeData(object):
        def __init__(self):
            self.phrase = None
            self.ID = None
            self.pinyin_sequence = None

    def __init__(self,vertex_count,pinyin_sequences,dictionary):
        print "Making lexicon graph..."
        super(LexiconGraph,self).__init__()
        self.vertex_count = vertex_count
        self.InitializeVertex()

        for pinyin_sequence in pinyin_sequences:
            phrases = dictionary.GetPhrases(pinyin_sequence.Pinyins())
            if phrases:
                for phrase in phrases:
                    e = self.AddEdge(self.vertex[pinyin_sequence.f],self.vertex[pinyin_sequence.to])
                    e.data = self.EdgeData()
                    e.data.pinyin_sequence = pinyin_sequence
                    e.data.phrase = phrase
                    e.data.ID = self.edge_count

        print "Done"


class Distance(object):
    def __init__(self,distance=None,previous=None):
        self.distance = distance
        self.previous = previous


class DistanceSet(object):
    def __init__(self,limit):
        self.limit = limit
        self.dlist = list()

    def Add(self,new_dist):
        added = False
        
        num = 0
        for current in self.dlist:
            if new_dist.distance > current.distance:
                self.dlist.insert(num,new_dist)
                if len(self.dlist) > self.limit:
                    self.dlist.pop(len(self.dlist)-1)
                added = True
                break
            num += 1
        if not added:
            if len(self.dlist) == self.limit:
                return False
            self.dlist.append(new_dist)
        return True

class SLMGraph(Graph):


    class EdgeData(object):
        def __init__(self):
            self.weight = 0


    class VertexData(object):
        def __init__(self,solution_set_size):
            self.phrase = None
            self.pinyin_sequence = None
            self.calculated = False
            self.dist = DistanceSet(solution_set_size)
    

    def __init__(self,lexicon_graph,slm,solution_set_size):
        print "Making Statistical Language Model Graph..."
        self.phrase_seq = list()
        self.sentences = list()
        super(SLMGraph,self).__init__()
        self.solution_set_size = solution_set_size
        self.vertex_count = lexicon_graph.edge_count + 2
        self.InitializeVertex()

        for i in range(self.vertex_count):
            self.vertex[i].data = self.VertexData(self.solution_set_size)
       
        for e in lexicon_graph.edges:
            ID = e.data.ID
            self.vertex[ID].data.phrase = e.data.phrase
            self.vertex[ID].data.pinyin_sequence = e.data.pinyin_sequence

        for e in lexicon_graph.vertex[0].to:
            ID = e.data.ID
            self.AddEdge(0,ID,slm.GetUnigram(e.data.phrase))

        for e in lexicon_graph.vertex[lexicon_graph.vertex_count - 1].f:
            ID = e.data.ID
            self.AddEdge(ID,self.vertex_count-1,1)

        self.vertex[0].data.phrase = "(S)"
        self.vertex[self.vertex_count-1].data.phrase = "(T)"

        for i in range(lexicon_graph.vertex_count):
            for eprev in lexicon_graph.vertex[i].f:
                prev_id = eprev.data.ID
                prev_phrase = eprev.data.phrase
                for esucc in lexicon_graph.vertex[i].to:
                    succ_id = esucc.data.ID
                    succ_phrase = esucc.data.phrase
                    self.AddEdge(prev_id,succ_id,slm.GetBigram(prev_phrase,succ_phrase))

        print("Done")

    def SolutionCompare(self,a,b):
        if a.probability > b.probability:
            return -1
        return 1
    
    def AddEdge(self,f,to,weight):
        e = super(SLMGraph,self).AddEdge(self.vertex[f],self.vertex[to])
        e.data = self.EdgeData()
        e.data.weight = self.CalculateWeight(weight)
        return e

    def CalculateWeight(self,weight):
        return math.log(weight)

    def CalculatePath(self,vcur):
        vcur_data = vcur.data
        vcur_data.calculated = True
        
        for e in vcur.f:
            weight = e.data.weight
            vprev = e.f
            vprev_data = vprev.data
            if not vprev_data.calculated:
                self.CalculatePath(vprev)

            for vprev_dist in vprev_data.dist.dlist:
                retval = vcur_data.dist.Add(Distance(vprev_dist.distance + weight,e))
                if not retval:
                    break

    def MakeSentence(self,*args):
        if not args:
            vs_data = self.vertex[0].data
            vs_data.calculated = True
            vs_data.dist.Add(Distance(0))

            self.CalculatePath(self.vertex[self.vertex_count-1])
            self.MakeSentence(self.vertex[self.vertex_count-1],0)
            
            self.sentences.sort(self.SolutionCompare)
            final_sentences = list()
            
            for i in range(len(self.sentences)):
                duplicated = False
                for solution_i in final_sentences:
                    if self.sentences[i].sentence == solution_i.sentence:
                        duplicated = True
                        break

                if duplicated:
                    continue

                final_sentences.append(self.sentences[i])
                if len(final_sentences) > self.solution_set_size:
                    break
            
            return final_sentences
        else:
            vcur = args[0]
            new_probability = args[1]
            self.phrase_seq.insert(0,vcur.data)

            for vdist in vcur.data.dist.dlist:
                prev = vdist.previous
                if prev:
                    self.MakeSentence(prev.f,new_probability+prev.data.weight)
                else:
                    pinyin = list()
                    sentence = ""
                    for i in self.phrase_seq:
                        if not i.phrase or not i.pinyin_sequence:
                            continue
                        sentence += i.phrase
                        for pinyin_segment in i.pinyin_sequence.Pinyins():
                            pinyin.append(pinyin_segment)
                    new_sentence = sentence
                    new_solution = PinYinModel.Solution()
                    new_solution.sentence = new_sentence
                    new_solution.probability = new_probability
                    self.sentences.append(new_solution)
            self.phrase_seq.pop(0)

class PinYinModel(object):
    class Solution(object):
        def __init__(self):
            self.sentence = None
            self.pinyin = list()
            self.probability = None

        def getPinyinString(self):
            return "".join(self.pinyin)

    def __init__(self):
        self.syllables = Syllable("syllable.bdt")
        self.dictionary = Lexicon("lexicon.bdt",self.syllables)
        self.slm = LanguageModel("slm.bdt",self.dictionary)
        self.syllable_graph = None
        self.lexicon_graph = None
        self.slm_graph = None

    def ParsePinyinString(self,string):
        self.syllable_graph = SyllableGraph(string,self.syllables)

    def MakeLexiconGraph(self):
        self.lexicon_graph = LexiconGraph(self.syllable_graph.vertex_count,self.syllable_graph.GetPinYinSequences(self.dictionary.max_length),self.dictionary)

    def MakeSLMGraph(self,solution_size):
        self.slm_graph = SLMGraph(self.lexicon_graph,self.slm,solution_size)

    def MakeSentences(self):
        return self.slm_graph.MakeSentence()



def main():
    pym = PinYinModel()
    string = "suzhoudaxue"
    pym.ParsePinyinString(string)
    pym.MakeLexiconGraph()
    pym.MakeSLMGraph(3)
    a = pym.MakeSentences()
    for i in a:
        print i.sentence
        print i.probability


if __name__ == "__main__":
    with Timer() as t:
        main()
    print t.interval
