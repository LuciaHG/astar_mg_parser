package edin.general

import java.io.File

object Global {

  /////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////      FINDING PROJECT DIRECTORY  ////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////

  val homeDir: String = System.getProperty("user.home")

  val projectDir: String = {

    val classDir:String = getClass.getProtectionDomain.getCodeSource.getLocation.getPath
    // jar file or classes dir

    var dir = new File(classDir)

    if(dir.isFile)
      dir = dir.getParentFile

    while( ! dir.list().contains("lib")){
      dir = dir.getParentFile
      if(dir == null)
        sys.error(s"parent dirs of $classDir don't contain 'lib' subdir")
    }
    dir.getAbsolutePath
  }


}

