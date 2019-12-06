/*
 *  SortByHdrs < infile.H > outfile.H keys="[+-]key1,[+-]key2,...,[+-]keyn" verbose=n synch=n
 */

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <locale.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <signal.h>
#include <arpa/inet.h>

#define MY_SEP_DOC \
sep_add_doc_line("NAME");\
sep_add_doc_line("    SortByHdrs - sort by sep3d header keys");\
sep_add_doc_line("");\
sep_add_doc_line("SYNOPSIS");\
sep_add_doc_line("    SortByHdrs < in.H > out.H keys=\"[+-]key1,[+-]key2,...,[+-]keyN\" synch=n verbose=n");\
sep_add_doc_line("");\
sep_add_doc_line("INPUT PARAMETERS");\
sep_add_doc_line("    keys - string");\
sep_add_doc_line("                  Comma-separated list of header key names");\
sep_add_doc_line("                  with optional + or - to indicate ascending");\
sep_add_doc_line("                  or descending sort for that key.");\
sep_add_doc_line("");\
sep_add_doc_line("    synch - boolean");\
sep_add_doc_line("                  Set to y or 1 to force trace reordering, not just headers.");\
sep_add_doc_line("");\
sep_add_doc_line("    verbose - boolean");\
sep_add_doc_line("                  Set to y or 1 for progress details.");\
sep_add_doc_line("");\
sep_add_doc_line("DESCRIPTION");\
sep_add_doc_line("    Sort up to 2 billion sep3d headers and traces using limited memory");\
sep_add_doc_line("    while making no assumptions on gridding.");\
sep_add_doc_line("    Selected headers must be scalars and may be a mix of integer and float.");\
sep_add_doc_line("    Temporary disk files are managed in the user datapath.");\
sep_add_doc_line("");\
sep_add_doc_line("SEE ALSO");\
sep_add_doc_line("    Sort3d");\
sep_add_doc_line("");\
sep_add_doc_line("CATEGORY");\
sep_add_doc_line("    util/cube");\
sep_add_doc_line(""); 

#include <sep.main> 

#include <sep3d.h> 

static volatile int wrapup;
static void gotowrapup(int signum) {
  wrapup = 1;
} 

int MAIN(void) {

  int n1; 
  float *intrace;
  int *inhdr;
  int verbose = 0;
  int synch = 0;
  char keybuffer[BUFSIZ];
  char datadir[BUFSIZ];
  char sortcmd[BUFSIZ];
  char tempbuf[BUFSIZ]; 
  char hdrtype[32];
  int nbytes, nreed, nrite;
  int nbytesHdr;
  int kindex;
  size_t ikey, nkeys; 
  int nvals = 1;
  int j,k;
  char **keylist;
#define INTKEY 1
#define FLOATKEY 2
  int  *keytype;
  char *cptr;
  char **ccptr;
  char ctmp;
  size_t slen;
  int irec, nrec;
  int jrec;
  char *sortdir;
  union {
    int i;
    float f;
  } convbuf; 
  int imgood;
  int itemp;
  int istat;
  int drn_key;
  int drn;
  int ioff;
  int nhdrbytes;
  int nhdrblocks;
  char *tag_header[1];
  pid_t cpid;
  FILE *sortInput, *sortOutput;
  int sortinPipeFds[2];
  int sortoutPipeFds[2];
  char **arglist;
  int iarg = 0; 

  init_3d();

  (void) setlocale(LC_ALL,"C");
  if(0 == getch("sync synch synchronize","1",&synch)) synch = 0;
  if(0 == getch("verb verbose","1",&verbose)) verbose = 0;
  (void) datapath(datadir);
  if(verbose) fprintf(stderr,"datadir=\"%s\"\n",datadir);
  if(0 == fetch("n1","d",&n1)) { seperr("Missing n1!\n"); }
  
  keybuffer[0] = '\0';
  if(0 == getch("keys","s",keybuffer)) {
       seperr("Missing parameter keys!\n");
  }
  if(keybuffer[0] == '\0') {
       seperr("Missing parameter keys!\n");
  }

       
  /* count number of comma separated keys */
  nkeys = 1;
  for(cptr = keybuffer; (*cptr) != '\0'; ++cptr) {
    if( (*cptr) == ',') ++nkeys;
  }

  if(verbose) fprintf(stderr,"number of keys is %lu\n",(unsigned long) nkeys);
  /* capture each key location in the list */
  /* and turn the commas into null terminators */
  keylist = (char **) calloc(sizeof(char *),nkeys);
  keytype = (int *) calloc(sizeof(int),nkeys);
  sortdir = (char *) calloc(sizeof(char), nkeys+1);
  if(keylist == ((char **)NULL) || keytype == ((int *)NULL)
     || sortdir == ((char *)NULL) ) {
     seperr("Unable to allocate memory!\n");
  }

  cptr = keybuffer;
  ikey = 0;
  keylist[ikey] = cptr;
  while((*cptr) != '\0') {
     if((*cptr) == ',') {
       *cptr = '\0';
       ++cptr;
       ++ikey;
       keylist[ikey] = cptr;
     }
     ++cptr;
  }

  /* strip any leading white space in key names */
  for(ikey = 0; ikey<nkeys; ++ikey) {
     while(isspace(keylist[ikey][0])) ++(keylist[ikey]);
  }
     
  /* strip any trailing white space in key names */
  for(ikey = 0; ikey<nkeys; ++ikey) {
     slen = strlen(keylist[ikey]);
     while(slen > 0) {
        slen--;
        if(! isspace(keylist[ikey][slen]) ) break;
        keylist[ikey][slen]='\0';
     }
  }

  /* handle sort directions, explicit or implicit */
  for(ikey=0; ikey<nkeys; ++ikey) {
     sortdir[ikey]= '+';
     ctmp = keylist[ikey][0];
     if(ctmp == '+' || ctmp == '-') {
        sortdir[ikey] = ctmp;
        ++(keylist[ikey]);
     }
  }

  if(verbose) {
     for(ikey=0; ikey<nkeys; ++ikey) {
       fprintf(stderr,"key%lu is \"%s\" sort direction %c\n",(unsigned long)ikey+1,keylist[ikey],sortdir[ikey]);
     }
  } 

  /* ensure keys are legitimate */
  for(ikey = 0; ikey < nkeys; ++ikey) {
     kindex = -1;
     (void) sep_get_key_index("in", keylist[ikey], &kindex);
     if(kindex < 0) {
        seperr("key \"%s\" not in header!\n",keylist[ikey]);
     }
     hdrtype[0] = '\0';
     (void) sep_get_key_type("in", &kindex, hdrtype);
     if(0 == strcmp(hdrtype,"scalar_int")) keytype[ikey] = INTKEY;
     if(0 == strcmp(hdrtype,"scalar_float")) keytype[ikey] = FLOATKEY;
     if(0 == keytype[ikey]) {
        seperr("key \"%s\" not scalar int or float!\n",keylist[ikey]);
     }
  }

  if(0 != sep_get_header_format_tag("in",tag_header)) {
     seperr("Error finding header tag\n");
  }
     
  if(0 != sep_get_header_bytes("in",&nhdrbytes)) {
     seperr("Error finding header length\n");
  }

  nhdrblocks = ssize_block(*tag_header,nhdrbytes);
  
  if(verbose) fprintf(stderr,"number of header bytes is %d\n",nhdrbytes);
  if(verbose) fprintf(stderr,"number of headers is %d\n",nhdrblocks);

  free(*tag_header); 
  

  if(0 != sep_copy_hff("in","out")) {
     seperr("Problem setting up output header keys\n");
  }
  /* handle data_record_number as needed */
  istat=sep_get_number_keys("in",&itemp);
  if(istat != 0) synch = 1;
  if(synch) {
     istat = sep_get_key_index("in", "data_record_number",&drn_key);
     if( istat != 0 ) {
        drn_key = -1;
     }
     if(istat == 2) { /* create new data_record_number header */
        itemp++;
        sep_put_number_keys("out",&itemp);
        sep_put_key("out","data_record_number","scalar_int","xdr_int",&itemp);
        drn_key = itemp;
     }
  } 
  /* copy over grid file if present */
  if(0 != sep_copy_gff("in","out")) {
     itemp = -1;
     putch("gff","d",&itemp);
  }

  /* If we're reordering headers and data, tell output to ignore */
  /* data record numbers, if present in headers.                 */
  itemp = synch ? 1 : 0;
  putch("same_record_number","d",&itemp);

  /* set up for and spawn Unix sort command */
  if(0 != pipe(sortinPipeFds)) {
      seperr("Unable to run pipe()\n");
  }
  if(0 != pipe(sortoutPipeFds)) {
      seperr("Unable to run pipe()\n");
  }

  /* set up to write to Unix sort */
  sortInput = fdopen(sortinPipeFds[1],"wb");

  if(verbose)
  fprintf(stderr,"sortinPipeFds (r=%d,w=%d) sortoutPipeFds (r=%d,w=%d)\n",
          sortinPipeFds[0],sortinPipeFds[1],
          sortoutPipeFds[0],sortoutPipeFds[1]); 
 
  /*hclose(); */ /* breaks sep_3d_close()!! */

  if(!synch) sep_copy_data_pointer("in","out"); 

  if(verbose) fprintf(stderr,"Finished history processing.\n");

  intrace = (float *)calloc(n1,sizeof(float));
  if(intrace == ((float *) NULL)) {
      seperr("Unable to allocate memory!\n");
  }
  nbytes = (int) (n1*sizeof(float));
  sep_get_header_bytes("in",&nbytesHdr);
  inhdr = (int *) calloc(nbytesHdr+sizeof(int)/*room for possible new drn*/,sizeof(char));
  if(inhdr == ((int *) NULL)) {
      seperr("Unable to allocate memory!\n");
  }

  arglist = (char **) calloc(nkeys*2+5+1,sizeof(char *));
  if(arglist == ((char **) NULL)) {
      seperr("Failed to allocate memory\n");
  }

  cpid = fork();
  if(cpid == -1) {
     seperr("fork() of Unix sort failed\n");
  }
  if(cpid == 0) { /* child launches Unix sort */
     if(dup2(sortinPipeFds[0],0) < 0) {
        seperr("Problem setting Unix sort input file descriptor.\n");
     }
     if(dup2(sortoutPipeFds[1],1) < 1) {
        seperr("Problem setting Unix sort output file descriptor.\n");
     }
     /* close pipe file discriptors we are not or no longer using */
     (void) close(sortinPipeFds[1]);
     (void) close(sortoutPipeFds[0]);
     (void) close(sortinPipeFds[0]);
     (void) close(sortoutPipeFds[1]);

     arglist[iarg++] = strdup("sort");
     arglist[iarg++] = strdup("-T");
     arglist[iarg++] = strdup(datadir);
     arglist[iarg++] = strdup("-S");
     arglist[iarg++] = strdup("25%");
     sprintf(sortcmd,"sort -T \"%s\" -S 25%% ",datadir);

     for(ikey=0; ikey<nkeys; ++ikey) {
       arglist[iarg++] = strdup("-k");
       sprintf(tempbuf,"%lu,%lun",(unsigned long)ikey+2,(unsigned long)ikey+2);
       if(sortdir[ikey] == '-') strcat(tempbuf,"r");
       arglist[iarg++] = strdup(tempbuf);
       strcat(sortcmd," -k ");
       strcat(sortcmd,tempbuf);
     }
     strcat(sortcmd," ");
     arglist[iarg++] = (char *) NULL;
     if(verbose) fprintf(stderr,"invoking sort command \"%s\"\n",sortcmd);

     (void) setenv("LC_ALL","C",1); /* protect against minus sign collation */
     istat = execvp("/bin/sort",arglist);

     if(istat == -1) {
        seperr("Failed to launch sort command \"%s\"\n",sortcmd);
     }
  } 

  (void) close(sortinPipeFds[0]);
  (void) close(sortoutPipeFds[1]); 

  wrapup = 0;
  (void) signal(SIGINT,gotowrapup); 

  /* scan all headers and write out a sort file */
  irec = 0;
  imgood = 1; 
  do {
     ++irec;
     if(wrapup) break;
     if(irec > nhdrblocks) {
           imgood = !imgood;
           break;
     }
     for(ikey=0; ikey<nkeys; ++ikey) {
        
        if(0 != sep_get_val_by_name("in",&irec,keylist[ikey],&nvals,&convbuf)){
        /* this doesn't work due to seplib bug that aborts instead */
           imgood = !imgood;
           break;
        }
        if(keytype[ikey] == FLOATKEY) {
           if(convbuf.f < 0.0) {
              convbuf.f = -convbuf.f;
              convbuf.i = -convbuf.i;
           }      
        }
        if(ikey == 0) fprintf(sortInput,"%d",irec-1);
        if(irec == nhdrblocks && verbose) fprintf(stderr,"%d",irec-1);
        fprintf(sortInput," %d ",convbuf.i);
        if(irec == nhdrblocks && verbose) fprintf(stderr," %d ",convbuf.i);
        if(ikey == (nkeys-1)) fprintf(sortInput,"\n");
        if(irec == nhdrblocks && verbose) fprintf(stderr,"\n");
     }
  } while(imgood);
  nrec = irec - 1;

  /* sent all input header key records to sort.  Close to signal EOF to sort */
  fclose(sortInput);

  if(wrapup) fprintf(stderr,"User interrupt - processing only first %d records.\n", (int) nrec);
  if(verbose) fprintf(stderr,"wrote %d records to Unix sort\n",(int) nrec);

  /* set up to read from Unix sort */
  sortOutput = fdopen(sortoutPipeFds[0],"rb");
  if(sortOutput == ((FILE *) NULL)) {
     perror("Failed to open Unix sort output\n");
     nrec = 0;
  }

  for(irec=0; irec<nrec; ) {
    if(NULL == fgets(tempbuf,sizeof(tempbuf),sortOutput)) {
       perror("SortByHdrs: reading sortOutput");
    }
    if(verbose && irec == 0) fprintf(stderr,"read first sorted record: %s",tempbuf);
    if(feof(sortOutput)) {
       seperr("Short output from Unix sort last record %d\n",irec);
    }
    if(ferror(sortOutput)) {
       seperr("I/O error reading from Unix sort last record %d\n",irec);
    }
       
    sscanf(tempbuf," %d",&jrec);
    jrec++;
    irec++;
    if(verbose && irec == nrec) fprintf(stderr,"read last sorted record: %s",tempbuf);
    
    istat = sep_get_val_headers("in",&jrec,&nvals,inhdr);
    if(istat != 0) {
       if(istat != 1) { /* we allow for old style SEP77 format which has no headers */
          seperr("Problem retrieving input headers. Exiting...");
       }
    }
    /* renumber data_record_number in header if reordering traces to match headers */
    if(synch && (drn_key > -1)) {
       drn = inhdr[drn_key-1];
       inhdr[drn_key-1] = irec;
    }
    /* write headers to output if available */
    if(istat == 0) {
       istat = sep_put_val_headers("out",&irec,&nvals,inhdr);
       if(istat != 0) {
          seperr("Problem writing output headers. Exiting...");
       }
    }
    if(synch) {
       if(drn_key < 0) drn = jrec;
       ioff = sseek_block("in",drn-1,n1*4,0);
       if(ioff == -1 && drn != jrec) {
           seperr("Trace seek problem, ioff=%d.  Exiting...",ioff);
       }
       nreed = sreed("in",intrace,n1*4);
       if(nreed != n1*4) {
           seperr("Trace read problem, nreed=%d.  Exiting...",nreed);
       }
       nrite = srite("out",intrace,n1*4);
       if(nrite != n1*4) {
           seperr("Trace write problem, nrite=%d.  Exiting...",nrite);
       }
    }
  } /* end of main R/W loop */

  if(sortOutput != ((FILE *) NULL)) {
     fclose(sortOutput);
     if(verbose) fprintf(stderr,"closed sortOutput\n");
  }
  waitpid(cpid,&istat,0);
  if(WIFEXITED(istat)) {
  if(verbose) fprintf(stderr,"checking sort exit status\n");
     if(WEXITSTATUS(istat) != EXIT_SUCCESS) {
       sepwarn(1,"sort command exit status %d??\n",WEXITSTATUS(istat));
     }
  } 
  (void) free(inhdr);
  (void) free(intrace);
  for(ccptr = arglist; (*ccptr); ++ccptr) {
     (void) free(*ccptr);
  }
  (void) free(arglist);
  (void) free(sortdir);
  (void) free(keytype);
  (void) free(keylist);

  if(verbose) fprintf(stderr,"freed buffers\n");

  sep_3d_close();

  if(verbose) fprintf(stderr,"closed sep3d\n");
  if(nrec > 0) {
    return (EXIT_SUCCESS);
  }
   return (EXIT_FAILURE);
}
