package edin.supertagger

import java.io.PrintWriter

import edin.algorithms.Zipper
import edin.nn.DynetSetup

import scala.io.Source

object MainApply {

  case class CMDargs(
                      model_dirs            : List[String] = null,
                      input_file_words      : String       = null,
                      input_file_aux_tags   : String       = null,
                      output_file           : String       = null,
                      output_file_best_k    : String       = null,
                      topK                  : Int          = 1,
                      topBeta               : Float        = -1f,
                      dynet_mem             : String       = null,
                      dynet_autobatch       : Int          = 0
                    )

  def main(args:Array[String]) : Unit = {
    val parser = new scopt.OptionParser[CMDargs](SUPERTAGGER_NAME) {
      head(SUPERTAGGER_NAME, SUPERTAGGER_VERSION.toString)
      opt[ Seq[String] ]( "model_dirs"            ).action((x,c) => c.copy( model_dirs            = x.toList )).required()
      opt[ String      ]( "input_file_words"      ).action((x,c) => c.copy( input_file_words      = x        )).required()
      opt[ String      ]( "input_file_aux_tags"   ).action((x,c) => c.copy( input_file_aux_tags   = x        ))
      opt[ String      ]( "output_file"           ).action((x,c) => c.copy( output_file           = x        )).required()
      opt[ String      ]( "output_file_best_k"    ).action((x,c) => c.copy( output_file_best_k    = x        ))
      opt[ Int         ]( "top_K"                 ).action((x,c) => c.copy( topK                  = x        ))
      opt[ Double      ]( "top_beta"              ).action((x,c) => c.copy( topBeta               = x.toFloat))
      opt[ Int         ]( "dynet-autobatch"       ).action((x,c) => c.copy( dynet_autobatch       = x        ))
      opt[ String      ]( "dynet-mem"             ).action((x,c) => c.copy( dynet_mem             = x        ))
      help("help").text("prints this usage text")
    }


    parser.parse(args, CMDargs()) match {
      case Some(cmd_args) =>

        if((cmd_args.topK>1 || cmd_args.topBeta>0.0) && cmd_args.output_file_best_k == null){
          System.err.println("If you want k-best you must also specify --output_file_best_k")
          System.exit(-1)
        }

        DynetSetup.init_dynet(
          cmd_args.dynet_mem,
          cmd_args.dynet_autobatch)

        val ensamble = new Ensamble(cmd_args.model_dirs)
        if(ensamble.modelContainers.head.hyperParams("main-vars").getOrElse("aux-tag-rep-dim", 0)>0 && cmd_args.input_file_aux_tags==null){
          System.err.println("missing the auxiliary tags file")
          System.exit(-1)
        }

        val outFh = new PrintWriter(cmd_args.output_file)
        val outBestKFh = if(cmd_args.output_file_best_k == null) null else new PrintWriter(cmd_args.output_file_best_k)

        val words_iterator = Source.fromFile(cmd_args.input_file_words).getLines().toIterable
        val auxTags_iterator = if(cmd_args.input_file_aux_tags != null){
          Source.fromFile(cmd_args.input_file_words).getLines().toIterable
        }else{
          Stream.from(1).map{_ => ""}
        }
        Zipper.zip3(words_iterator, auxTags_iterator, Stream.from(1)).foreach{ case (line, aux_line, lineId) =>
        // words_iterator.zipWithIndex.foreach{ case (line, lineId) =>
          if(lineId % 10 == 0){
            System.err.println(s"processing $lineId")
          }
          val words = line.split(" +").toList
          val auxs = if(aux_line == null || aux_line == "") words.map{_ => null} else aux_line.asInstanceOf[String].split(" +").toList
          val tags  = ensamble.predictBestTagSequence(words, auxs)
          outFh.println(tags.mkString(" "))
          if(outBestKFh != null){
            val tagssWithScores:List[List[(String, Double)]] = if(cmd_args.topBeta > 0){
              ensamble.predictBetaBestTagSequenceWithScores(words, auxs, cmd_args.topBeta)
            }else{
              ensamble.predictKBestTagSequenceWithScores(words, auxs, cmd_args.topK)
            }
            (words zip tagssWithScores).foreach{ case (word, tagsWithScores) =>
                val tagOut = tagsWithScores.map{case (tag, score) => s"$tag $score"}.mkString("\t")
                outBestKFh.println(s"$word\tX\t$tagOut")
            }
            outBestKFh.println()
          }
        }
        outFh.close()
        if(outBestKFh != null)
          outBestKFh.close()
      case None =>
        System.err.println("You didn't specify all the required arguments")
        System.exit(-1)
    }
  }

}
