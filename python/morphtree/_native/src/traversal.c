/* MorphTree-owned traversal ABI1:read an existing declarative tree only.
 * No training,split policy or program selection is encoded here.
 * Stable owned nonaliasing caller buffers;bounded traversal even for cycles.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <float.h>
_Static_assert(sizeof(size_t)==8 && sizeof(double)==8 && DBL_MANT_DIG==53,"64bit binary64 ABI required");
#if defined(_WIN32)
#define MT_EXPORT __declspec(dllexport)
#else
#define MT_EXPORT __attribute__((visibility("default")))
#endif
MT_EXPORT uint32_t mt_traversal_abi_version(void){return 1u;}
static uint32_t bin_at(const void *column,uint32_t width,uint64_t row){
    if(width==1u)return ((const uint8_t*)column)[row];
    if(width==2u){uint16_t v;memcpy(&v,(const uint8_t*)column+row*2u,2u);return v;}
    {uint32_t v;memcpy(&v,(const uint8_t*)column+row*4u,4u);return v;}
}
MT_EXPORT int mt_predict_tree_f64_v1(
    const void *const *columns,const uint32_t *widths,const uint8_t *nominal,
    uint64_t features,uint64_t rows,const uint8_t *leaf,const double *values,
    const int64_t *feature,const int64_t *cut,const uint8_t *missing_left,
    const int64_t *left,const int64_t *right,uint64_t nodes,double *out,uint64_t output_size){
    if(!features || !nodes || nodes>INT64_MAX || rows>SIZE_MAX/8u || output_size<rows
       || !columns || !widths || !nominal || !leaf || !values || !feature || !cut
       || !missing_left || !left || !right || (rows && !out))return 1;
    for(uint64_t f=0;f<features;f++){
        if((widths[f]!=1u && widths[f]!=2u && widths[f]!=4u) || nominal[f]>1u
           || rows>SIZE_MAX/widths[f] || (rows && !columns[f]))return 1;
    }
    /* Validate every index before any possible column/node read. */
    for(uint64_t k=0;k<nodes;k++){
        if(leaf[k]>1u || missing_left[k]>1u)return 1;
        if(!leaf[k] && (feature[k]<0 || (uint64_t)feature[k]>=features || cut[k]<0
           || (uint64_t)cut[k]>UINT32_MAX || left[k]<0 || right[k]<0
           || (uint64_t)left[k]>=nodes || (uint64_t)right[k]>=nodes))return 2;
    }
    for(uint64_t row=0;row<rows;row++){
        uint64_t k=0;int reached=0;
        for(uint64_t hop=0;hop<=nodes;hop++){
            if(leaf[k]){out[row]=values[k];reached=1;break;}
            uint64_t f=(uint64_t)feature[k];uint32_t b=bin_at(columns[f],widths[f],row);
            int go_left=nominal[f] ? b==(uint32_t)cut[k] : b==0u ? missing_left[k] : b<=(uint32_t)cut[k];
            k=(uint64_t)(go_left ? left[k] : right[k]);
        }
        if(!reached)return 3; /* Cyclic graph,no unbounded native loop. */
    }
    return 0;
}
