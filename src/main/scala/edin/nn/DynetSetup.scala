package edin.nn

import edu.cmu.dynet.internal.DynetParams
import edu.cmu.dynet.{ComputationGraph, internal}

object DynetSetup {

  def init_dynet(dynet_mem:String, autobatch:Int) : Unit = {
    val params:DynetParams = new internal.DynetParams()

    params.setAutobatch(autobatch)
    if(dynet_mem != null)
      params.setMem_descriptor(dynet_mem)

    internal.dynet_swig.initialize(params)
  }

  private var expressions = List[AnyRef]()
  def safeReference(x:AnyRef) : Unit = {
    expressions ::= x
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  private var cg_counter = 0

  def cg_id : Int = {
    cg_counter
  }

  def cg_renew() : Unit = {
    expressions = List()
    cg_counter += 1
    ComputationGraph.renew()
  }

}
