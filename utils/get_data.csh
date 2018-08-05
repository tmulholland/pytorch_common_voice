#!/bin/tcsh

wget https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz

tar -xvf cv_corpus_v1.tar.gz

setenv base cv_corpus_v1/cv-other-test
foreach mp3 ( `ls $base` )
   setenv wav `echo $mp3 | sed 's/mp3$/wav/'`
   mpg123 -w $base/$wav $base/$mp3
end

setenv base cv_corpus_v1/cv-other-train
foreach mp3 ( `ls $base` )
   setenv wav `echo $mp3 | sed 's/mp3$/wav/'`
   mpg123 -w $base/$wav $base/$mp3
end
